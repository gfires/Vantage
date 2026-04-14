# gstack

For all web browsing tasks, use the `/browse` skill from gstack. Never use `mcp__claude-in-chrome__*` tools.

## Available Skills

- `/office-hours` - Office hours workflow
- `/plan-ceo-review` - Plan CEO review
- `/plan-eng-review` - Plan engineering review
- `/plan-design-review` - Plan design review
- `/design-consultation` - Design consultation
- `/design-shotgun` - Design shotgun
- `/design-html` - Design HTML
- `/review` - Code review
- `/ship` - Ship workflow
- `/land-and-deploy` - Land and deploy
- `/canary` - Canary deployment
- `/benchmark` - Benchmarking
- `/browse` - Web browsing
- `/connect-chrome` - Connect to Chrome
- `/qa` - QA testing
- `/qa-only` - QA only testing
- `/design-review` - Design review
- `/setup-browser-cookies` - Setup browser cookies
- `/setup-deploy` - Setup deployment
- `/retro` - Retrospective
- `/investigate` - Investigation
- `/document-release` - Document release
- `/codex` - Codex
- `/cso` - CSO workflow
- `/autoplan` - Auto planning
- `/plan-devex-review` - Plan DevEx review
- `/devex-review` - DevEx review
- `/careful` - Careful mode
- `/freeze` - Freeze workflow
- `/guard` - Guard mode
- `/unfreeze` - Unfreeze workflow
- `/gstack-upgrade` - Upgrade gstack
- `/learn` - Learning mode

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
- Save progress, checkpoint, resume → invoke checkpoint
- Code quality, health check → invoke health
