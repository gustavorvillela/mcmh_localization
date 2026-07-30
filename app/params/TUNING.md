# Tuning de parametros por modelo

Este projeto tem uma ferramenta para buscar bons parametros por modelo:

```bash
./app/scripts/tune_params.py
```

Ela gera YAMLs candidatos, roda `test_algs.launch`, reconstroi as metricas com
`offline_evaluate.py` e salva:

- `tuning_results/<run>/ranking.csv`
- `tuning_results/<run>/best_params/<MODE>.yaml`
- `tuning_results/<run>/params/<MODE>/*.yaml`

## Rodada pequena

Boa para testar se o fluxo esta funcionando:

```bash
./app/scripts/tune_params.py \
  --preset tiny \
  --max-candidates-per-mode 3 \
  --modes MCL MHMCL AMCL MHAMCL 3MCL \
  --bags explore_bin.bag \
  --repeats 3
```

## Rodada recomendada

Melhor para escolher parametros com mais seguranca:

```bash
./app/scripts/tune_params.py \
  --preset quick \
  --max-candidates-per-mode 12 \
  --modes MCL MHMCL AMCL MHAMCL 3MCL \
  --bags explore_bin.bag straight_line_spin.bag static.bag \
  --repeats 5
```

## Rodada maior

Use quando tiver tempo de processamento:

```bash
./app/scripts/tune_params.py \
  --preset full \
  --max-candidates-per-mode 30 \
  --modes MCL MHMCL AMCL MHAMCL 3MCL \
  --bags app/bags \
  --repeats 10
```

## Comparacao justa

Para comparar modelos sem deixar cada um alterar parametros fisicos do robo e
do sensor, use:

```bash
./app/scripts/tune_params.py --fair-physical
```

Isso fixa parametros como `alpha1-4`, `sigma_hit`, `z_hit`, `z_rand` e `step`,
e ajusta principalmente os parametros especificos de cada algoritmo.

## Ranqueamento sem repetir simulacoes

Se uma rodada ja existe:

```bash
./app/scripts/tune_params.py --rank-only tuning_results/<run>
```

## Perfis por modelo para benchmark

Os melhores parametros encontrados em `explore_bin_fair_budget_plus` foram
salvos como perfil medio por modelo:

- `mcl_medium.yaml`
- `mhmcl_medium.yaml`
- `amcl_medium.yaml`
- `mhamcl_medium.yaml`

Tambem existem variantes:

- `*_aggressive.yaml`: menor orcamento computacional
- `*_conservative.yaml`: maior orcamento/robustez

Os scripts `run_all_modes.sh` e `run_particle_sweep.sh` agora escolhem o YAML
pelo par modelo/perfil. Por exemplo:

- `MCL + M` usa `mcl_medium.yaml`
- `MHMCL + A` usa `mhmcl_aggressive.yaml`
- `AMCL + C` usa `amcl_conservative.yaml`
- `MHAMCL + M` usa `mhamcl_medium.yaml`

Para o benchmark principal do paper em `explore_bin.bag`, rode:

```bash
./app/scripts/run_all_modes.sh explore_bin.bag
```

Por padrao, esse comando usa apenas o perfil `M`, isto e, os melhores
parametros tunados por modelo. Para comparar os tres perfis, altere
`SCENARIOS=("M")` para `SCENARIOS=("C" "M" "A")` em `run_all_modes.sh`.

## Score usado

Por padrao, o score minimizado e:

```text
score =
  rmse_pos_mean
  + 0.2 * rmse_yaw_mean
  + 1.0 * (1 - success_rate)
  + 0.05 * failure_events_mean
  + 0.00005 * init_particles
```

Menor score significa melhor candidato. Os pesos podem ser alterados com:

```bash
--yaw-weight
--failure-penalty
--failure-event-weight
--failure-rate-weight
--particle-weight
```
