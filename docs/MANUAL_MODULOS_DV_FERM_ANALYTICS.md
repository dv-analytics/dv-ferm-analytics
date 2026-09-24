# Manual de utilização dos módulos `dv-ferm-analytics`

## 1. Objetivo

O pacote `dv-ferm-analytics` concentra rotinas reutilizáveis de análise industrial, benchmarking, árvore de valor, conferência e estimativa de RTC, regressões, correlações, tabelas comparativas e geração de relatórios Word.

A biblioteca pública deve conter **somente lógica genérica e reutilizável**. Dados de clientes, nomes/códigos de usinas, caminhos internos, preços corporativos, credenciais, templates privados e resultados reais devem permanecer fora do pacote, em projetos privados que consumam a biblioteca.

Este manual descreve o contrato de entrada e saída dos módulos atualmente usados e validados no projeto.

---

## 1.1 Licenciamento

O pacote é distribuído sob **licença proprietária/restrita**. A disponibilidade
pública do código-fonte ou do pacote não concede, por si só, autorização para
reutilização, modificação, redistribuição, sublicenciamento, comercialização ou
criação de trabalhos derivados. As condições aplicáveis estão no arquivo
`LICENSE` da raiz do projeto.

Os projetos consumidores devem garantir que possuem autorização de uso compatível
com a licença antes de instalar ou incorporar o pacote.

---

## 2. Requisitos gerais

### 2.1 Python e instalação

O projeto foi desenvolvido e testado no ambiente atual com:

- Python `>=3.13,<3.14`;
- Poetry 2.x para desenvolvimento e empacotamento;
- layout `src/`;
- import Python: `dv_ferm_analytics`;
- distribuição PyPI: `dv-ferm-analytics`.

Em desenvolvimento local:

```powershell
poetry install
poetry run pytest
```

Enquanto o pacote ainda estiver sendo usado localmente por outro projeto, pode ser instalado em modo editável apontando para a pasta do repositório. Depois da publicação no PyPI, a instalação esperada será:

```powershell
poetry add dv-ferm-analytics
```

ou:

```powershell
pip install dv-ferm-analytics
```

### 2.2 Dependências funcionais

As funcionalidades do pacote usam, conforme o módulo:

- `pandas`: manipulação de tabelas;
- `numpy`: cálculos numéricos;
- `scipy`: testes estatísticos;
- `scikit-learn`: regressões, imputação e detecção de outliers;
- `matplotlib`: geração de figuras;
- `python-docx`: geração de relatórios Word;
- `shap`: explicação local da regressão múltipla quando disponível.

`jinja2` é **opcional**. Ele só é necessário quando se deseja usar `pandas.Styler`, por exemplo através de `build_comparison_styler()`. Para relatórios Word, `jinja2` não é necessário.

Para `get_cepea_prices()` também é necessário acesso à internet e disponibilidade da fonte consultada. A rotina deve falhar de forma explícita quando não consegue obter os preços; não deve existir fallback silencioso com preço privado ou fixo.

### 2.3 Requisitos dos DataFrames

Há dois formatos principais de dados.

#### Formato longo (`long`)

É o formato usado pela maior parte dos módulos industriais. Cada linha representa um indicador em uma unidade/período.

Exemplo genérico:

```text
DATA_HORA  | tag_benchmarking | VALOR_MES | VALOR_ACUMULADO | usina_code_benchmarking
2026-01-31 | RTC%             | 91.20     | 91.20           | 999
2026-01-31 | EXTRTOTAL%       | 96.10     | 96.10           | 999
2026-02-28 | RTC%             | 91.80     | 91.50           | 999
```

Colunas recorrentes:

| Coluna | Tipo esperado | Uso |
|---|---|---|
| `DATA_HORA` | datetime ou conversível | seleção do último registro e período |
| `tag_benchmarking` | texto | código do indicador |
| `VALOR_ACUMULADO` | numérico | análises acumuladas |
| `VALOR_MES` | numérico | análises mensais e testes pareados |
| `ano` | inteiro | pode ser derivado de `DATA_HORA` |
| `mes` | inteiro 1–12 | pode ser derivado de `DATA_HORA` |
| `usina_nome` | texto | identificação no projeto consumidor |
| `usina_code_benchmarking` | inteiro | seleção de unidade/referências |
| `usina_destilaria_autonoma` | booleano | seleção de regras aplicáveis |
| `difusor` | booleano | seleção de população de benchmarking |

Quando há vários registros para a mesma combinação de ano/mês/indicador, os validadores privados normalmente ordenam por `DATA_HORA` e mantêm o registro mais recente.

#### Formato largo (`wide`)

É o formato usado pelas regressões e correlações. Cada linha representa um período e cada coluna representa uma variável.

```text
ano | mes | RTC% | EXTRTOTAL% | RENDGERDEST | PRDTORTAFILT
2025|  1  | 90.8 | 95.9       | 89.2        | 1.30
2025|  2  | 91.1 | 96.0       | 89.4        | 1.22
2026|  1  | 91.7 | 96.2       | 90.0        | 1.10
```

Os nomes das colunas dos indicadores devem ser consistentes entre os períodos. Valores numéricos inválidos são convertidos para `NaN` pelos módulos quando aplicável.

---

## 3. Benchmarking Top N

### Importação

```python
from dv_ferm_analytics.analysis import calculate_top_benchmark
```

### Objetivo

Selecionar as unidades de melhor desempenho no RTC e construir a referência média dos indicadores das unidades selecionadas.

### Entrada principal

`df_peers`: DataFrame longo contendo a população candidata ao benchmarking. É boa prática filtrar antes:

- o período de interesse;
- o tipo de unidade compatível;
- a própria unidade analisada, para que ela não componha o benchmark.

No fluxo GAOA, o DataFrame contém pelo menos os identificadores da unidade, `tag_benchmarking`, `VALOR_ACUMULADO` e data/período.

### Exemplo

```python
top = calculate_top_benchmark(
    df_peers,
    autonoma=False,
    difusor=False,
    top_n=5,
)
```

### Saída

O objeto retornado é usado principalmente através de:

- `top.top_units`: ranking das unidades de referência;
- `top.indicator_means`: médias dos indicadores das unidades selecionadas, prontas para uso em `calculate_value_tree()`;
- metadados de referência, quando expostos pela versão instalada, como a data de referência.

### Cuidados

- não inclua a própria unidade na população do benchmark, salvo se esse for explicitamente o objetivo;
- use apenas unidades comparáveis quanto à configuração industrial;
- não misture períodos diferentes sem uma regra explícita de corte temporal.

---

## 4. Preços CEPEA

### Importação

```python
from dv_ferm_analytics.market import CepeaFetchError, get_cepea_prices
```

### Objetivo

Obter os preços de referência usados na monetização da árvore de valor.

### Exemplo

```python
try:
    prices = get_cepea_prices()
except CepeaFetchError as exc:
    raise RuntimeError("Não foi possível obter os preços de mercado") from exc

print(prices.sugar_bag_brl)
print(prices.ethanol_liter_brl)
```

### Saída esperada

Objeto compatível com `CommodityPrices`, contendo:

- `sugar_bag_brl`;
- `ethanol_liter_brl`;
- `sugar_reference` quando disponível;
- `ethanol_reference` quando disponível.

### Requisitos

- internet;
- fonte CEPEA disponível;
- validação de faixa/consistência deve ser feita no projeto consumidor quando necessário.

---

## 5. Árvore de valor — benchmark externo

### Importação

```python
from dv_ferm_analytics.analysis.value_tree import (
    CommodityPrices,
    calculate_value_tree,
)
```

### Objetivo

Comparar a unidade com uma referência externa, normalmente a média Top 5, e estimar oportunidades econômicas por indicador.

### Entradas

#### `df_usina`

DataFrame longo da unidade analisada. Deve conter, no mínimo:

- `tag_benchmarking`;
- `VALOR_ACUMULADO`.

É recomendado também possuir `DATA_HORA`, porque o módulo usa o último valor válido quando existem várias observações por indicador.

#### `df_top5`

Pode ser fornecido em dois formatos:

1. tabela já agregada com uma coluna `media_top`; ou
2. dados brutos das unidades de referência com `VALOR_ACUMULADO`, para que a média seja calculada.

#### `autonoma`

Booleano que define o conjunto de indicadores aplicável à unidade.

#### `commodity_prices`

```python
prices = CommodityPrices(
    sugar_bag_brl=100.0,       # valor apenas ilustrativo
    ethanol_liter_brl=2.0,     # valor apenas ilustrativo
    sugar_reference="exemplo",
    ethanol_reference="exemplo",
)
```

#### `input_prices`

Dicionário opcional de preços de insumos. As chaves devem ser tags de indicadores, por exemplo:

```python
input_prices = {
    "CALGSC": 0.001,
    "PLIMGPSC": 0.020,
    "ACIDOSULF": 0.005,
    "ANTESPDISPDA": 0.010,
}
```

Os valores acima são apenas exemplos sintéticos; preços corporativos reais não devem ser colocados no pacote público.

#### Entradas industriais opcionais

- `moagem_safra_ton`;
- `art_cana_medio_pct`;
- `sugar_mix_pct`.

Quando `None`, o módulo tenta resolver os valores a partir das tags padrão:

- moagem: `CANAPROCES`;
- ART da cana: `ARTCANAGIDES`;
- mix: `ARTENSAC`.

### Exemplo

```python
result = calculate_value_tree(
    df_usina,
    df_top5,
    autonoma=False,
    commodity_prices=prices,
    input_prices=input_prices,
)
```

### Saída — `ValueTreeResult`

- `result.table`: tabela principal;
- `result.assumptions`: premissas e memória textual dos cálculos;
- `result.total_potential_gain_brl`: soma dos potenciais financeiros válidos.

A tabela principal contém:

```text
Setor
Parâmetro
tag_benchmarking
Usina
TOP_05
Diferença (p.p.)
Potencial de ganho em R$
```

### Regra de ganho mínimo

Na implementação atual existe `GANHO_MINIMO = 30000`. Ganhos calculados abaixo ou iguais a esse limite não são tratados como oportunidade financeira válida pela regra correspondente. Se esse limiar for alterado futuramente, a documentação e os testes devem ser atualizados em conjunto.

---

## 6. Árvore de valor — melhor ano histórico

### Importação

```python
from dv_ferm_analytics.analysis.value_tree_best_year import (
    calculate_best_year_value_tree,
)
```

### Objetivo

Comparar o período atual com o **ano de maior RTC da própria unidade para o mesmo mês**.

### Entrada

DataFrame longo histórico da mesma unidade, contendo:

- `DATA_HORA`;
- `tag_benchmarking`;
- `VALOR_ACUMULADO`.

Também são necessários os mesmos parâmetros econômicos da árvore de valor base.

### Exemplo

```python
result = calculate_best_year_value_tree(
    df_usina_historico,
    current_year=2026,
    current_month=8,
    autonoma=False,
    commodity_prices=prices,
    input_prices=input_prices,
    include_current_year_in_reference=True,
)
```

### Saída — `BestYearValueTreeResult`

Além da árvore monetizada:

- `table`;
- `assumptions`;
- `total_potential_gain_brl`;
- `current_year`, `current_month`;
- `reference_year`;
- `current_rtc`, `reference_rtc`;
- `reference_date`;
- `rtc_history`.

Na tabela, a comparação é apresentada como `Ano atual` versus `Melhor ano`.

---

## 7. Árvore de valor — ano perfeito sintético

### Importação

```python
from dv_ferm_analytics.analysis.value_tree_perfect_year import (
    calculate_perfect_year_value_tree,
)
```

### Objetivo

Construir uma referência histórica sintética em que **cada indicador pode vir de um ano diferente**, selecionando o melhor valor histórico daquele indicador para o mesmo mês.

### Exemplo

```python
result = calculate_perfect_year_value_tree(
    df_usina_historico,
    current_year=2026,
    current_month=8,
    autonoma=False,
    commodity_prices=prices,
    input_prices=input_prices,
    include_current_year_in_reference=True,
    include_mix=True,
)
```

### Saída — `PerfectYearValueTreeResult`

- `table`;
- `assumptions`;
- `total_potential_gain_brl`;
- `current_year`, `current_month`, `current_rtc`;
- `include_current_year_in_reference`;
- `include_mix`;
- `reference_indicators`: indicador, valor e ano/data de referência escolhidos;
- `indicator_history`: histórico considerado na seleção;
- `rtc_history`.

### Atenção

O resultado é um benchmark sintético. Não representa necessariamente uma safra histórica que tenha ocorrido de forma integral em um único ano.

---

## 8. Árvore de valor — usinas de referência escolhidas

### Importação

```python
from dv_ferm_analytics.analysis.value_tree_selected_units import (
    calculate_selected_units_value_tree,
)
```

### Objetivo

Comparar uma unidade-alvo com uma lista explícita de unidades de referência, sem depender do Top 5 por RTC.

### Entrada adicional

O DataFrame deve conter:

- `usina_code_benchmarking`;
- `tag_benchmarking`;
- `VALOR_ACUMULADO`;
- `DATA_HORA`.

### Exemplo

```python
result = calculate_selected_units_value_tree(
    df,
    target_unit_code=999,
    reference_unit_codes=[101, 102, 103],
    current_year=2026,
    current_month=8,
    autonoma=False,
    commodity_prices=prices,
    input_prices=input_prices,
    include_mix=True,
    require_all_reference_units=True,
)
```

Os códigos acima são apenas exemplos fictícios.

### Saída — `SelectedUnitsValueTreeResult`

- `table`;
- `assumptions`;
- `total_potential_gain_brl`;
- `target_unit_code`;
- `reference_unit_codes`;
- `missing_reference_unit_codes`;
- `reference_units`;
- `indicator_means`;
- `reference_detail`.

Quando `require_all_reference_units=True`, a ausência de uma unidade de referência solicitada deve ser tratada como erro.

---

## 9. Árvore de valor — ano anterior

### Importação

```python
from dv_ferm_analytics.analysis.value_tree_previous_year import (
    calculate_previous_year_value_tree,
)
```

### Objetivo

Comparar o mês atual da unidade com o **mesmo mês do ano imediatamente anterior**.

### Exemplo

```python
result = calculate_previous_year_value_tree(
    df_usina_historico,
    current_year=2026,
    current_month=8,
    autonoma=False,
    commodity_prices=prices,
    input_prices=input_prices,
    include_mix_in_analysis=True,
)
```

### Saída — `PreviousYearValueTreeResult`

- `table`;
- `assumptions`;
- `total_potential_gain_brl`;
- `current_year`, `previous_year`, `current_month`;
- `current_rtc`, `previous_rtc`;
- `current_snapshot`;
- `previous_year_snapshot`.

A tabela é apresentada como `Ano atual` versus `Ano anterior`.

---

## 10. Conferência do RTC

### Importação

```python
from dv_ferm_analytics.analysis.rtc_conference import (
    RtcConferenceIndicators,
    calculate_rtc_conference,
)
```

### Objetivo

Conferir o RTC entre dois anos e decompor diferenças relacionadas a perdas, extração, RGD, mix e produtividades.

### Entrada

DataFrame longo com:

- `tag_benchmarking`;
- `VALOR_ACUMULADO`;
- `DATA_HORA`, **ou** as colunas `ano` e `mes`.

O módulo filtra o mês solicitado e usa o último valor acumulado válido de cada indicador em cada ano.

### Exemplo

```python
result = calculate_rtc_conference(
    df_ams,
    current_year=2026,
    previous_year=2025,
    month=8,
)
```

É possível sobrescrever tags e descrições com `RtcConferenceIndicators`.

### Saída — `RtcConferenceResult`

- `part1_table`: tabela formatada da decomposição principal;
- `part1_raw`: mesma tabela com valores numéricos;
- `part2_table`: comparação relativa de RTC, ART da cana e produtividades;
- `part2_raw`: versão numérica;
- `rtc_previous`, `rtc_current`;
- `theoretical_rtc`;
- `rtc_vs_theoretical_pp`;
- `current_mix_pct`;
- `weighted_productivity_difference_pct`;
- `rtc_plus_art_difference_pct`;
- `notes`.

---

## 11. Estimativa indireta de RTC, RGD e perdas indeterminadas

### Importação

```python
from dv_ferm_analytics.analysis.rtc_indirect_estimation import (
    calculate_indirect_rtc,
)
```

### Objetivo

Reproduzir a metodologia de estimativa indireta de RGD, perdas indeterminadas e RTC a partir de indicadores industriais.

### Entrada

O DataFrame deve estar **previamente filtrado para uma única unidade e o período desejado**. O mínimo estrutural é:

- `tag_benchmarking`;
- `VALOR_ACUMULADO`;
- `DATA_HORA` opcional, mas recomendado para resolver o último valor válido.

Entre as tags usadas pela metodologia estão, conforme disponibilidade:

`PRDBAGACO`, `PRDTORTAFILT`, `PRDAGLAVCANA`, `PRDMULTIGER`, `PRDAGRESGER`, `PRDINDETERM`, `RTC%`, `EXTRTOTAL%`, `ARTENSAC`, `RENDGERDEST`, `BASTVINH10^5`, `VINHO%ALCOOL`, `PRDVINHFLEGM`, `ARTVMO`, `TEMPVINBRMAX`, `BIOM%ARTMOST`, `GLIC%ARTMOST` e `RECETCO2`.

### Exemplo

```python
result = calculate_indirect_rtc(
    df_periodo_usina,
    indicator_col="tag_benchmarking",
    value_col="VALOR_ACUMULADO",
    date_col="DATA_HORA",
)
```

### Saída — `IndirectRtcResult`

- `summary_table`: observado, estimado e diferença para RTC, RGD e perdas indeterminadas;
- `calculation_table`: memória de cálculo detalhada;
- `input_table`: valores de entrada utilizados;
- `diagnostics_table`: verificações auxiliares;
- valores escalares observados e estimados;
- `autonomous`;
- `mix_pct_original` e `mix_pct_used`.

---

## 12. Regressão linear múltipla do RTC

### Importação

```python
from dv_ferm_analytics.analysis.rtc_multiple_linear_regression import (
    calculate_rtc_multiple_linear_regression,
)
```

### Objetivo

Modelar `RTC%` como variável dependente de vários indicadores industriais e produzir diagnóstico global, explicação local e simulações com o ano anterior.

### Formato de entrada

Este módulo usa **duas tabelas wide**.

#### `abt_mensal`

- uma linha por ano/mês;
- colunas `ano`, `mes`;
- uma coluna por indicador;
- valores provenientes de `VALOR_MES`.

#### `abt_acumulado`

- mesma estrutura de linhas e indicadores;
- valores provenientes de `VALOR_ACUMULADO`;
- usada para os snapshots do ano atual e do ano anterior.

Exemplo de construção a partir de dados long:

```python
latest = (
    df_long.sort_values("DATA_HORA")
    .drop_duplicates(["ano", "mes", "tag_benchmarking"], keep="last")
)

abt_mensal = (
    latest.pivot_table(
        index=["ano", "mes"],
        columns="tag_benchmarking",
        values="VALOR_MES",
        aggfunc="last",
    )
    .reset_index()
)

abt_acumulado = (
    latest.pivot_table(
        index=["ano", "mes"],
        columns="tag_benchmarking",
        values="VALOR_ACUMULADO",
        aggfunc="last",
    )
    .reset_index()
)
```

### Exemplo

```python
result = calculate_rtc_multiple_linear_regression(
    abt_mensal,
    abt_acumulado,
    current_year=2026,
    previous_year=2025,
    autonomous=False,
    target="RTC%",
    current_month=8,
    min_r2=0.40,
)
```

### Requisitos e regras importantes

- o target deve existir e possuir pelo menos `min_valid_target` valores válidos;
- colunas com excesso de ausentes ou invariantes podem ser excluídas;
- dados faltantes dos preditores são tratados por KNN quando aplicável;
- a seleção de variáveis usa forward selection por R² ajustado;
- restrições de sinal retiram relações incompatíveis com a direção industrial esperada;
- o R² reportado segue a semântica do código legado: ajuste sobre o conjunto de treinamento utilizado;
- se `R² < min_r2`, as saídas locais/simulações são suprimidas;
- `shap` é usado para a explicação local quando disponível; caso contrário existe fallback por decomposição linear.

### Saída — `RTCLinearRegressionResult`

Figuras em memória (`BytesIO`):

- `image_shap`: gráfico de coeficientes padronizados; o nome foi preservado por compatibilidade histórica;
- `image_explainer`: explicação local, preferencialmente waterfall SHAP;
- `box`: comparação do valor atual com a distribuição de treinamento.

Tabelas e textos:

- `df_simulation`;
- `texto_explicativo`;
- `texto_retorno`;
- `texto_simulation`;
- `coefficients`;
- `local_contributions`;
- `training_table`;
- `correlation_matrix`;
- `diagnostics_table`;
- `selected_features`, `removed_sign_constraints`, `excluded_features`;
- `model_ok`, `r2`, `n_observations`.

As contribuições do modelo explicam **a predição**, não demonstram causalidade.

---

## 13. Correlação e regressão linear simples

### Importação

```python
from dv_ferm_analytics.analysis.correlation_simple_regression import (
    calculate_simple_correlation_regression,
)
```

### Objetivo

Analisar várias variáveis contra um único target por correlação de Pearson e regressão linear simples, com remoção opcional de outliers e regressão quadrática exploratória opcional.

### Entrada

DataFrame wide, com:

- coluna do `target`;
- colunas das `features`;
- opcionalmente uma coluna temporal usada no painel de evolução.

### Exemplo

```python
features = [c for c in df_wide.columns if c not in {"ano", "mes", "RTC%"}]

result = calculate_simple_correlation_regression(
    df_wide,
    target="RTC%",
    features=features,
    r_min=0.40,
    r_max=0.97,
    time_col="periodo",
    include_quadratic=False,
    min_observations=4,
    alpha=0.05,
    remove_outliers=True,
    outlier_method="IsolationForest",
    contamination=0.10,
    outlier_score_z_limit=2.0,
)
```

### Seleção das relações

A seleção final usa:

```text
r_min <= |r de Pearson| < r_max
```

além do número mínimo de observações.

A remoção histórica de outliers usa o score do detector e exclui pontos cujo `|z-score do score|` ultrapassa o limite configurado.

Métodos disponíveis:

- `IsolationForest`;
- `LocalOutlierFactor`;
- `EllipticEnvelope`.

### Saída — `SimpleCorrelationRegressionResult`

- `summary_table`: somente relações selecionadas;
- `all_pairs_table`: todas as relações calculadas;
- `figures`: tupla de `CorrelationRegressionFigure`;
- `image_list`: atalho para as imagens;
- `figure_map`: dicionário `feature -> BytesIO`;
- `ordered_features`: features ordenadas pela análise;
- `outlier_table`: auditoria dos pontos removidos;
- `excluded_features`: variáveis não analisadas e motivo;
- `diagnostics_table`;
- `text_summary`.

A `summary_table` inclui `n`, `r`, `|r|`, R² linear, p-valores, coeficiente angular, intercepto, R² quadrático e quantidade de outliers removidos.

Correlação e regressão simples indicam associação; não devem ser interpretadas isoladamente como causalidade.

---

## 14. Tabelas comparativas entre safras

### Importação

```python
from dv_ferm_analytics.analysis.comparative_tables import (
    ComparisonIndicator,
    calculate_year_comparison,
)
```

### Objetivo

Comparar valores acumulados do ano atual e do ano anterior e calcular significância estatística usando `VALOR_MES` pareado por mês.

### Entrada

DataFrame longo contendo obrigatoriamente:

- `tag_benchmarking`;
- `ano`;
- `mes`;
- `VALOR_ACUMULADO`;
- `VALOR_MES`.

`DATA_HORA` é recomendado para desempate do valor acumulado quando houver mais de um registro no mesmo período.

### Definição dos indicadores

Use `ComparisonIndicator` sempre que quiser controlar explicitamente nome e direção:

```python
indicators = [
    ComparisonIndicator("RTC%", label="RTC", direction="higher"),
    ComparisonIndicator(
        "PRDTORTAFILT",
        label="Perda torta de filtro (%)",
        direction="lower",
    ),
]
```

Direções:

- `higher`: quanto maior, melhor;
- `lower`: quanto menor, melhor;
- `neutral`: sem interpretação de desempenho.

### Exemplo

```python
result = calculate_year_comparison(
    df,
    indicators,
    current_year=2026,
    previous_year=2025,
    current_month=8,
    alpha=0.05,
    normality_alpha=0.05,
    min_pairs=4,
)
```

### Cálculos

```text
Diferença = Valor acumulado atual - Valor acumulado anterior
```

```text
Diferença relativa (%) = Diferença / Valor acumulado anterior * 100
```

Quando o denominador é zero ou ausente, a diferença relativa é `NaN`.

Para a significância:

1. os valores de `VALOR_MES` são pareados pelo número do mês;
2. calcula-se a diferença mensal entre os anos;
3. Shapiro-Wilk testa a normalidade das diferenças;
4. se a normalidade não for rejeitada, usa-se teste t pareado;
5. caso contrário, usa-se Wilcoxon pareado.

### Regra de cores

Para `higher`:

- diferença positiva → azul;
- diferença negativa → vermelho.

Para `lower`:

- diferença negativa → azul;
- diferença positiva → vermelho.

A cor não é misturada ao DataFrame principal. Ela é entregue em `style_table` para a camada de apresentação.

### Saída — `ComparativeTableResult`

- `table`: tabela final com seis colunas;
- `audit_table`: teste usado, p-valores, n de pares, direção e cor;
- `paired_values`: pares mensais efetivamente usados;
- `style_table`: metadados para formatação;
- `difference_colors`: atalho com as cores na ordem da tabela;
- `current_year`, `previous_year`, `reference_month`.

Tabela principal:

```text
Indicador
Valor Acumulado <ano anterior>
Valor Acumulado <ano atual>
Diferença
Diferença relativa (%)
Significância
```

### `build_comparison_styler()`

É um helper opcional para HTML/Excel e requer `jinja2`. Para Word, use `result.table` + `result.style_table`.

---

## 15. Geração de relatórios Word

### Importação

```python
from dv_ferm_analytics.word import WordReport
```

### Objetivo

Centralizar a criação de documentos Word com título, cabeçalho, rodapé, tabelas e figuras.

### Requisitos

- `python-docx`;
- template `.docx` quando for necessário preservar o padrão corporativo;
- caminho de saída gravável.

### Exemplo básico

```python
report = WordReport(template_path="template.docx")

report.set_header_fields(
    usina="Unidade exemplo",
    mes="Agosto/2026",
)

report.set_footer(
    document_code="CODIGO",
    show_page_number=True,
)

report.add_title("Relatório")
report.add_heading("Resultados", level=1)
report.add_paragraph("Texto explicativo.")
report.add_data_table(df_resultado, include_index=False, decimal_places=2)
report.add_figure(figura_ou_buffer, width_cm=16.0)
report.save("relatorio.docx")
```

`add_figure()` aceita figuras Matplotlib, caminhos de imagem e streams `BytesIO` compatíveis.

Templates corporativos e cabeçalhos privados **não devem ser distribuídos dentro do pacote público**.

---

## 16. Fluxos recomendados

### 16.1 Benchmarking + árvore de valor

```text
DataFrame AMS long
    ↓ filtrar período
    ↓ excluir unidade-alvo da população de peers
calculate_top_benchmark()
    ↓ indicator_means
calculate_value_tree()
    ↓
ValueTreeResult
    ↓
WordReport / CSV / gráficos
```

### 16.2 Regressão múltipla

```text
DataFrame AMS long
    ↓ último registro por ano/mês/tag
    ├─ pivot VALOR_MES      -> abt_mensal
    └─ pivot VALOR_ACUMULADO -> abt_acumulado
    ↓
calculate_rtc_multiple_linear_regression()
```

### 16.3 Correlação simples

```text
DataFrame AMS long
    ↓ pivot VALOR_MES
DataFrame wide
    ↓
calculate_simple_correlation_regression()
```

### 16.4 Comparação anual

```text
DataFrame AMS long com VALOR_MES + VALOR_ACUMULADO
    ↓
calculate_year_comparison()
    ├─ table
    ├─ audit_table
    ├─ paired_values
    └─ style_table
```

---

## 17. Testes e validação antes de publicar

Antes de cada release:

```powershell
poetry run pytest
```

Também é recomendado:

```powershell
poetry check
poetry build
```

A suíte pública deve usar apenas dados sintéticos. Testes não devem depender de:

- bancos privados;
- credenciais;
- caminhos locais específicos;
- arquivos corporativos;
- nomes/códigos reais de clientes;
- preços privados.

---

## 18. Checklist de diagnóstico de dados

Antes de chamar qualquer módulo:

1. confirme que a unidade/período foi filtrada corretamente;
2. converta `DATA_HORA` para datetime;
3. converta valores numéricos com `pd.to_numeric(..., errors="coerce")`;
4. verifique duplicidades por ano/mês/tag;
5. confirme se as tags esperadas existem;
6. não transforme ausência de dados em zero sem que a metodologia determine isso;
7. para testes pareados, confirme que há meses correspondentes nos dois anos;
8. para regressões, confirme número mínimo de observações e variância das colunas;
9. para benchmarking, retire a própria unidade da população quando necessário;
10. mantenha a separação entre lógica pública e configuração privada.

---

## 19. Política de segurança dos dados

O repositório público/PyPI deve permanecer genérico. Não publicar:

- nomes e códigos reais de usinas;
- caminhos internos de rede ou de projetos privados;
- `HEROKU_DATABASE_URL` ou qualquer credencial;
- `.env`;
- preços corporativos de insumos;
- arquivos AMS reais;
- templates Word corporativos;
- relatórios gerados para clientes.

A camada privada (`gaoa_analytics_reports`) é responsável por conectar os módulos genéricos aos dados reais, templates e configurações corporativas.
