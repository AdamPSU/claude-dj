# AI Hackathon 2026 tracks knowledge base

Source mix: AI Hackathon site, Kandra Chau sponsor emails, user-provided sponsor context, and public company pages where noted. Email-derived sponsor criteria are treated as the strongest signal.

## Event baseline

- Site: https://ai.hackberkeley.org/
- Format: 24-hour, in-person UC Berkeley AI Hackathon, June 20-21, 2026.
- Scale: 1,300+ hackers, 300+ projects, $100,000 in prizes.
- Build rule: ideation before the event is allowed, but implementation must happen during the hacking period.
- Team size: up to 4.
- Core positioning: LLMs, open-source APIs, AI tools, and projects with clear impact or strong execution.

## General tracks

### Ddoski's World

Social impact track. Good fit for EdTech, civic tech, environmental tools, access, equity, and projects motivated by a real-world social challenge.

Judging signal: seriousness of the problem, usefulness to affected users, and how clearly the technology addresses the social challenge.

### Ddoski's Toolbox

Tools for developers, creators, and knowledge workers. Good fit for developer utilities, CLI tools, APIs, automation scripts, workflow apps, project management systems, writing aids, design aids, and productivity software.

Judging signal: usefulness, usability, and execution quality. The site explicitly says the tool itself matters more than just the idea.

### Ddoski's Lab

Science and engineering track. Good fit for health tech, medical tools, hardware prototypes, embedded systems, biotech, data-driven research tools, and hard engineering problems.

Judging signal: technical depth plus real-world application. Software, hardware, and hybrid projects all fit.

### Ddoski's Playground

Creative and experimental track. Good fit for games, interactive experiences, generative art, humor-driven hacks, experimental interfaces, and unconventional ideas.

Judging signal: originality, execution, and how compelling the experience is. The project does not need to solve a serious problem.

## Sponsor tracks and sponsor fit

### Fetch AI

Track shape: agentic web.

What they are building: infrastructure for autonomous agents that can discover each other, coordinate, collaborate, and complete real-world tasks. Key primitives mentioned in email: ASI:One, Agentverse, Fetch Business, Chat Protocol, and Payment Protocol.

What they seem to value:

- Discoverable agents, not private demos.
- Agents that coordinate with other agents or services.
- Real user or business tasks.
- A working lifecycle: create, deploy, register, expose, and monetize.
- Practical use of ASI:One and Agentverse.

Strong project angle: a useful agent or agent network that solves a concrete workflow, appears in Agentverse, can be reached through ASI:One, and has a clear user-facing task.

Evidence: organizer email, Fetch workshop notes, Fetch public pages.

### Annapurna Labs

Track shape: cloud AI infrastructure and custom silicon.

What they are building: custom silicon for AWS, including Graviton processors and Trainium and Inferentia ML accelerators.

What they seem to value:

- Hardware-software depth.
- Efficient AI training and inference.
- Cloud infrastructure awareness.
- Strong systems thinking.
- Recruiting fit for hardware, ASIC, and software engineering.

Strong project angle: a performance-aware AI system, inference optimization tool, benchmarking harness, model-serving improvement, hardware-adjacent dev tool, or anything that shows awareness of cost, latency, throughput, or accelerator constraints.

Evidence: organizer email, AWS and Amazon public pages. The email did not include explicit prize judging criteria.

### Redis

Track shape: real-time context for AI apps.

What they are building: fast data and context infrastructure for AI applications, with vector search, caching, and memory.

What they are looking for:

- Redis beyond caching.
- Redis Iris for agent memory, vector search, and context retrieval.
- Creativity and originality, especially solving human problems in fresh or fun ways.
- Technical implementation quality, including correctness, scalability, and architecture.

Strong project angle: an AI assistant or workflow tool that uses Redis for RAG, semantic caching, agent memory, fast context retrieval, or stateful multi-turn behavior. Show latency or quality gains if possible.

Evidence: organizer email, Redis public docs/pages.

### Claude / Anthropic

Track shape: ambitious Claude Code builds.

What they are looking for: projects built with Claude Code that tackle meaningful issues in health, education, economic opportunity, or another domain where AI can help people.

What they seem to value:

- Taking a big swing at a hard problem.
- Social usefulness over polish alone.
- Aspiration and effort, even if the 24-hour build is incomplete.
- Accessibility for non-CS students and teams learning as they go.
- Responsible, human-centered use of AI.

Strong project angle: a Claude Code-built system aimed at a hard human problem. The pitch should explain why the problem matters, what a real user can do with it, and what was hard about the build.

Evidence: organizer email, Anthropic public pages.

### Deepgram

Track shape: voice AI and audio interfaces.

What they are building: speech and voice APIs, including Aura Text-to-Speech for generating spoken audio from text.

What they seem to value:

- Working voice/audio integrations, not just text-only demos.
- Low-latency narration or conversational voice experiences.
- Clear product moments where speech improves the user experience.
- Thoughtful voice choice and output quality.

Strong project angle: use Deepgram as the DJ narration layer. Claude decides what to say, the MCP `narrate` tool turns it into a short spoken DJ line, and the mascot app can later show the same line as text. Preferred voice direction is an African-American DJ-style voice/persona if a suitable Deepgram voice is available; exact Aura voice/model is TBD after auditioning and API verification.

Evidence: user-provided sponsor context, Deepgram public pages, and Deepgram developer docs for Aura Text-to-Speech voices and REST/streaming APIs.

### The Interaction Company

Track shape: personal AI assistants and agentic automation.

What they are building: Poke, an AI assistant that knows the user and handles calendar, email, daily tasks, coordination, and admin work in the background.

What they are looking for:

- Technical depth of the integration.
- Useful automation.
- Originality.
- Third-party miniapps built with their API.
- Incoming triggers connected to complex actions across the web.
- Code turned into agentic tools that many users can use.

Strong project angle: a personal or team automation that reacts to events, connects to real tools, executes multi-step tasks, and saves users coordination work.

Evidence: organizer email, public coverage of Poke and The Interaction Company.

### The Token Company

Track shape: context optimization and compression.

What they are building: custom context-compression models that cut token costs while improving downstream LLM behavior.

What they are looking for:

- Depth of research.
- Ingenuity.
- Creativity.
- SOTA context optimization.
- Demos that show compression, cost reduction, or smarter context use.

Strong project angle: a system that makes LLM context cheaper, shorter, faster, or more useful without hurting output quality. Best pitch includes evals: token savings, latency, cost, and quality before vs. after.

Evidence: organizer email, The Token Company public pages.

## Cross-sponsor strategy

- Best broad overlap: an AI assistant or agentic workflow with strong memory, retrieval, automation, and measurable context efficiency.
- Best technical story: Redis for memory/retrieval, Deepgram for spoken narration, Token Company-style context optimization, Fetch-style agent discoverability, and Claude Code for fast build velocity.
- Best impact story: choose a hard user problem that fits Ddoski's World, Toolbox, or Lab, then use sponsor tech as the implementation engine.
- Best demo posture: show a real workflow end to end. Sponsors repeatedly reward working integration, useful automation, technical depth, and clear problem fit.

## Source notes

- AI Hackathon site: general tracks, event format, sponsors, logistics.
- Organizer emails from Kandra Chau: sponsor track language for Fetch AI, Annapurna Labs, Redis, Claude, The Interaction Company, and The Token Company.
- Public company pages: used only to sharpen company-value context when the email did not state judging criteria directly.
