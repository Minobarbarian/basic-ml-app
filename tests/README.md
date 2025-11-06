# Tests

Para executar todos os testes:
```bash
# Todos os testes
python -m pytest -v

# Apenas testes de unidade e sanidade
python -m pytest -v -m "not integration"

# Apenas testes de integração
```

## --- Fixtures (Contextos de Teste) ---
paths()
clf_wandb(paths)
clf_local_trained(paths)
clf_minimal()
clf_with_stopwords(tmp_path)

## --- Testes de Unidade (Rápidos) ---
test_init_fails_without_config_or_model(monkeypatch)
test_preprocess_text_lowercase(clf_minimal)
test_preprocess_text_min_words(clf_minimal)
test_preprocess_text_stopwords(clf_with_stopwords)

## --- Testes de Sanidade Local (Médios) ---
test_local_train_model_created(clf_local_trained)
test_local_predict_sanity(clf_local_trained)
test_one_hot_encoder_local(clf_local_trained)

## --- Testes de Integração (Lentos) ---
test_wandb_model_predicts(clf_wandb)
test_wandb_model_accuracy_easy_examples(clf_wandb, paths)