# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Initial Bedrock Runtime GenAI instrumentation package.
- Support for `bedrock-runtime` `Converse` and `ConverseStream` LLM spans.
- Provider-aware support for `InvokeModel` and `InvokeModelWithResponseStream`.
- Bedrock Runtime message, token, tool definition, streaming, and TTFC extraction.
- InvokeModel extraction for Amazon Titan, Amazon Nova, Anthropic Claude, Cohere,
  Meta Llama, and Mistral JSON shapes.
- Example showing Bedrock Runtime instrumentation by itself and composed with
  AgentCore instrumentation.
- AgentCore composition example requirements for the Bedrock AgentCore SDK,
  Botocore CRT login support, and compatible OTel package versions.

### Fixed
- Align Bedrock spans with semantic conventions by setting
  `gen_ai.provider.name` to `aws.bedrock`.
- Normalize Bedrock and provider-specific finish reasons before emitting
  `gen_ai.response.finish_reasons`.
- Always populate invocation message and tool-call content on Python objects;
  GenAI emitters control whether content is exported.
- Avoid JSON parsing for obvious non-JSON tool-input strings in streaming
  parsing paths.
- Update the manual example dependency, content-capture, AgentCore ADOT, and
  span-parenting documentation.
