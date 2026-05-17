# Technical Specification: Kesoku Message Flow & Persistence Redesign (V3 - Concurrency & Logic)

## 1. The Pure Broker Pattern
- **Gateway**: A stateless hub using `post` and `listen`. 
- **Chatbots**: Decentralized subscribers. They handle their own delivery and offline recovery via `listen`.

## 2. Advanced Concurrency: Session Workers
To handle multiple users and "user interruptions" gracefully:

### A. The Agent Dispatcher
- The Agent runs a master `listen(role='user')` loop.
- Upon receiving a message, it checks if a `SessionWorker` task already exists for that `session_id`.
- If not, it spawns a dedicated `SessionWorker(session_id)` task.
- If it exists, it simply pushes the new message into that worker's internal queue.

### B. The Session Worker Logic (The "Anti-Stall" Mechanism)
Each worker processes its queue **one step at a time**:
1. **Pull**: Get the latest user message(s) from its queue.
2. **Step**: Perform ONE atomic action (either an LLM inference OR a Tool execution).
3. **Check-in**: After the step finishes, check the queue again.
4. **Re-evaluate**: If a new user message arrived while the tool was running, append it to the history and let the LLM decide if it should continue the previous task or pivot.
5. **Yield**: If the queue is empty and the Agent has reached a 'final response' state, the task can hibernate or terminate.

## 3. Tool Interruption Policy
- **Never Kill Mid-Tool**: For safety, tools are treated as atomic. We wait for the tool to return before looking at new user input.
- **Thought Interruption**: If the Agent is in a long "Thinking/Chain of Thought" loop, it can be interrupted between LLM calls.

## 4. Message Schema Updates
- `status`: `pending`, `processing`, `completed`, `interrupted`.
- Use `parent_id` to link tool results to specific tool calls, allowing the Agent to "re-align" after an interruption.

## 5. Verification Criteria
- **Multi-user**: User A and User B can use tools simultaneously.
- **Interruption**: Send "Calculate 1+1", then immediately send "Actually, calculate 2+2". The Agent should finish the first check-in, see the second message, and eventually provide the result for 2+2 (or both).
