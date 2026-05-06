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
