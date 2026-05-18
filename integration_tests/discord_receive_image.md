# Integration test: Test receiving messages in Discord

# Steps
Use configuration file: `private/config.toml`

- Start the service using `kesoku -c private/config.toml`.
- Open browser and navigate to the channel defined in `private/docs/discord.md`.
- Send a simple message to ask the agent to generate an image: "Please generate an image of a cat."
- Wait for response
- If there is a response, the test succeeds.
- If response doesn't appear within 300 seconds, please check your logs.
- Remember to shutdown the service after the test.