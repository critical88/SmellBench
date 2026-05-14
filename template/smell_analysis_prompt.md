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

<rubric>
<name>short criterion name, e.g. Core Smell Resolution</name>
<description>what this criterion evaluates, tied to the specific smell instance</description>
<excellent>9-10: what an excellent result looks like</excellent>
<good>7-8: what a good result looks like</good>
<acceptable>5-6: what an acceptable result looks like</acceptable>
<below_average>3-4: what a below-average result looks like</below_average>
<poor>0-2: what a poor result looks like</poor>
</rubric>

<rubric>
<name>second criterion name</name>
<description>what this criterion evaluates</description>
<excellent>9-10: description</excellent>
<good>7-8: description</good>
<acceptable>5-6: description</acceptable>
<below_average>3-4: description</below_average>
<poor>0-2: description</poor>
</rubric>

The two rubrics should capture the most important evaluation dimensions **specific to this particular smell instance** — things that the generic rubric would miss. They will be added as extra scoring criteria when evaluating candidate fixes. Make them concrete and tied to the actual function/class names in the diff.

Each rubric MUST include all 5 scoring levels. Each level should clearly describe what a result at that score range looks like, with concrete references to the code in the diff.
