"""Boss orchestrator — coordinates Actor and Critic.

A tiny state machine:

    actor_call → critic_review → decide
                                  ├── accept                     → return
                                  ├── needs_iteration + budget   → re-call actor with feedback
                                  └── reject / budget exhausted  → synthesize fallback

The Boss makes no LLM calls of its own, so every decision is traceable in
``BossTrace.iterations`` and reproducible from the inputs. The actor is a callable
``actor(critic_feedback | None) -> EstimationResult`` so iteration N+1 can receive
the feedback from iteration N. Wiring lives in ``EstimationService.estimate_with_acb``.
"""

from __future__ import annotations

from collections.abc import Callable

import structlog

from app.schemas.acb import ACBIteration, BossDecision, BossTrace
from app.schemas.critic import CriticFeedback
from app.schemas.estimation import LOW_CONFIDENCE_THRESHOLD, EstimationResult

log = structlog.get_logger()

ActorCallable = Callable[[CriticFeedback | None], EstimationResult]
CriticCallable = Callable[[EstimationResult], CriticFeedback]

SUMMARY_MAX_CHARS = 1200


class Boss:
    """Stateless orchestrator. One ``run`` per estimation."""

    def __init__(self, *, max_iterations: int = 3) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.max_iterations = max_iterations

    def run(
        self, *, actor: ActorCallable, critic: CriticCallable
    ) -> tuple[EstimationResult, BossTrace]:
        trace = BossTrace(final_decision="accept", iterations_run=0)
        feedback: CriticFeedback | None = None
        current: EstimationResult | None = None

        for iteration in range(self.max_iterations):
            current = actor(feedback)
            review = critic(current)
            decision = self._decide(review, iterations_left=self.max_iterations - iteration - 1)
            trace.iterations.append(
                ACBIteration(
                    iteration=iteration,
                    decision_after=decision,
                    critic_verdict=review.verdict,
                    critic_confidence=review.confidence_in_review,
                    issue_summary=[
                        f"[{issue.severity}] {issue.category} @ {issue.field_path}"
                        for issue in review.issues[:5]
                    ],
                )
            )
            trace.iterations_run = iteration + 1
            log.info(
                "boss_iteration",
                iteration=iteration,
                decision=decision,
                verdict=review.verdict,
                issues=len(review.issues),
            )

            if decision == "accept":
                trace.final_decision = "accept"
                return current, trace
            if decision == "synthesize":
                trace.final_decision = "synthesize"
                return self._synthesize_fallback(current, review), trace
            feedback = review

        # Unreachable while _decide never returns "iterate" on the last round; kept as a guard.
        assert current is not None
        trace.final_decision = "synthesize"
        log.info("boss_iteration_budget_exhausted", iterations=self.max_iterations)
        return self._synthesize_fallback(current, feedback), trace

    @staticmethod
    def _decide(review: CriticFeedback, *, iterations_left: int) -> BossDecision:
        if review.verdict == "accept":
            return "accept"
        if review.verdict == "reject":
            return "synthesize"
        return "iterate" if iterations_left > 0 else "synthesize"

    @staticmethod
    def _synthesize_fallback(
        last_result: EstimationResult, last_feedback: CriticFeedback | None
    ) -> EstimationResult:
        """Return the last draft annotated with the open issues and reduced confidence.

        An estimation with caveats serves the user better than a zeroed placeholder; the
        non-convergence signal already travels in ``BossTrace.final_decision``.
        """
        if last_feedback is None or not last_feedback.issues:
            caveats = "⚠ Open caveats from independent review: (no detail available)\n\n"
        else:
            caveats = (
                "⚠ Open caveats from independent review (loop did not fully converge):\n"
                + "\n".join(
                    f"- [{issue.severity}] {issue.category} ({issue.field_path}): {issue.description}"
                    for issue in last_feedback.issues
                )
                + "\n\n"
            )
        # Below the threshold the schema would demand an "Out of scope:" prefix.
        reduced_confidence = max(LOW_CONFIDENCE_THRESHOLD, last_result.confidence_pct // 2)
        return last_result.model_copy(
            update={
                "summary": (caveats + last_result.summary)[:SUMMARY_MAX_CHARS],
                "confidence_pct": reduced_confidence,
            }
        )
