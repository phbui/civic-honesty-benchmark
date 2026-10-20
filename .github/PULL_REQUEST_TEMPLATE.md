**What this changes and why**

**How it was tested**

The test command and its summary line. For a scorer or driver change, say which released run you rescored and whether any number moved.

**Checklist**

- [ ] Tests pass: `python -m pytest scripts -q`
- [ ] No API key, paid run, or live-API call is added to the default path
- [ ] No dataset content is bundled; questions, gold, and logs stay on Hugging Face
- [ ] A change to scoring or to a driver default says which published numbers it could move
