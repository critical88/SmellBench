You are an expert software engineer analyzing a code change (diff) that introduces a "{smell_type}" code smell into a codebase.

## Smell Type
- **Name**: {smell_type}
- **Description**: {smell_description}

## The Smell Diff
This diff was applied to a clean codebase to introduce the smell:
```diff
{smell_diff}
```

## Your Task
Analyze this diff by going through each change and assessing its significance. Focus on **what each change means and how important it is**, NOT on how to fix it.

For each distinct change in the diff (a new function, a moved block, an added import, etc.), explain:
- **What it does**
- **How significant it is** (critical / moderate / minor) to the smell
- **What it degrades** in the codebase (e.g., coupling, cohesion, readability, API surface, etc.)

After covering individual changes, provide:

1. **Overall smell pattern**: Summarize how these changes work together to create the "{smell_type}" smell. What design principle is violated?
2. **Severity ranking**: Rank the changes from most to least important. Which changes are the **root cause** of the smell, and which are just supporting noise?
3. **What was degraded overall**: What concrete qualities of the codebase were harmed? Be specific about the impact on maintainability, coupling, cohesion, etc.
4. **Key evaluation signals**: When judging whether a candidate fix truly addresses this smell, what should matter most? What would distinguish a thorough fix from a superficial one?

## Output Format

Return your result using XML tags. Do NOT wrap the output in a code block.

<analysis>
Your full analysis text as described above. Write freely — no escaping needed.
</analysis>
