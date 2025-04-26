# Thoughts:

- [ ] Y.js logic is beyond me atm. Should probably do simple client/server model and then figure out local first sync another time.

- [ ] When adding tasks to a column other than "open" in the mobile view, they appear to go straight to open anyway.

- [ ] Kanban columns and overall board need to be wider for desktop. Maybe implement fade, blurring at the edges instead of a hard cut off.

- [ ] Drag and drop is not very effective. Drag handle should be whole of the card. If users want to select text, they can edit the task and copy from the input field.

- [ ] Styling defaults. Why are scrollbars gross? Why are dropdowns gross?

- [ ] Syllable validation could use a supabase serverless function? There's a Python ML project that does this with 95% accuracy. Not sure if overkill, but apparently syllables are hard. Otherwise try a library that's more robust than our simple approach. Or go heavy handed and literally use an LLM like Claude Haiku. If we do go with a more general approach, we may be able to implement real haiku ratings in the validation! 6/10. Not your best, but ok.