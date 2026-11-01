Rails.application.config.estimator_ai = ActiveSupport::OrderedOptions.new.tap do |c|
  c.base_url = ENV.fetch("ESTIMATOR_API_BASE_URL", "http://localhost:8000")
  c.timeout  = ENV.fetch("ESTIMATOR_AI_TIMEOUT", "180").to_i
end
