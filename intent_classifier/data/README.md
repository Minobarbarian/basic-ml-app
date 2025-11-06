# data

Os dados de treino usados pelo `intent-classifier` devem estar no formato yml. 

Exemplo:

```
- intent: confusion
  examples:
    - wait what?
    - huh? im confused
    ...

- intent: neutral
  examples:
    - Alright, let's see what happens
    - I'm not ready
    - We should continue with the next part
    ...
```