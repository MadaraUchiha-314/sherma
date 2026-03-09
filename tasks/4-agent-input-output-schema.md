- Agents sometimes require custom input types conforming to a schema (apart from text messages)
- Agents sometimes also need to output custom types conforming to a schema (apart from text messages)
- We will use `DataPart` from A2A for this.
- Messages can contain a list of `Part`
- We will use `DataPart` with custom metadata to convey input and output which conform to a schema

- Agents also need to publish these input and output schema

- Thoroughly study the a2a protocol specification and see what are the avenues for doing this

- I found publishing this as an A2A extension might be the way to go.
    - But just consider this as another option and do your research and present an option.

A2A protocol: https://a2a-protocol.org/latest/specification/
- Read the web-page thoroughly