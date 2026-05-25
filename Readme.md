Use this command to add this repo as a submodule to your current repository:

```
git submodule add https://github.com/RealityShifts/Srivastava-book-of-Blocks.git models/blocks
```

then

```
git submodule update --init --recursive
```


### Diagrams

Architecture diagrams (one Mermaid `flowchart TD` per public block, 122 across 17 categories) live in a sibling repo so the model code here stays clean:

[**RealityShifts/Srivastava-book-of-Blocks-diagrams**](https://github.com/RealityShifts/Srivastava-book-of-Blocks-diagrams)

Specs live in [`_generate.py`](https://github.com/RealityShifts/Srivastava-book-of-Blocks-diagrams/blob/main/_generate.py); regenerate everything with `python _generate.py`.

