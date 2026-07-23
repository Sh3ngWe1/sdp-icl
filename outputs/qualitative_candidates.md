# SDP-ICL Qualitative Candidates Report

> 本報告由 `scripts/find_qualitative_cases.py` 自動產生。
> 所有案例均來自真實實驗結果，請人工審查後決定是否放入論文。

## 1. Success Cases（成功案例，F1 ≥ 0.8 或 EM = 1）

### Case 1 (SUCCESS)

**自動判斷原因**：SDP-ICL F1=1.000 (EM=1); 優於或持平 Baseline-Standard (F1=1.000); 優於 Baseline-Sanitized (F1=1.000)


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `5730a4d02461fd1900a9cf2b` |
| Model             | `qwen`                     |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.6926 sec/query`         |
| Answer Diversity  | `0.02`                     |


**Question**: Where did France focus its efforts to rebuild its empire?

**Ground Truth**: `Africa | North and West Africa | North and West Africa | North and West Africa, as well as South-East Asia, with other conquests in Central and East Africa, as well as the South Pacific | North and West Africa, as well as South-East Asia,`

**Context Snippet**:

> France took control of Algeria in 1830 but began in earnest to rebuild its worldwide empire after 1850, concentrating chiefly in North and West Africa, as well as South-East Asia, with other conquests in Central and East Africa, as well as the South Pacific. Republicans, at first hostile to empire, ...

**Answers**:


| Method                         | Answer                                            | F1    | EM  |
| ------------------------------ | ------------------------------------------------- | ----- | --- |
| Baseline-Standard (No Privacy) | North and West Africa, as well as South-East Asia | 1.000 | 1   |
| Baseline-Sanitized (DP-ICL)    | North and West Africa, as well as South-East Asia | 1.000 | 1   |
| **SDP-ICL (Proposed)**         | north and west africa as well as southeast asia   | 1.000 | 1   |


**SDP Ensemble Top Answers** (N=100 total): "north and west africa as well as southeast asia" × 99 | "north and west africa southeast asia" × 1

---

### Case 2 (SUCCESS)

**自動判斷原因**：SDP-ICL F1=1.000 (EM=1); 優於或持平 Baseline-Standard (F1=1.000); 優於 Baseline-Sanitized (F1=1.000)


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `57111ab8a58dae1900cd6c3c` |
| Model             | `qwen`                     |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.8865 sec/query`         |
| Answer Diversity  | `0.01`                     |


**Question**: What protestant religions made Northern European counties safe for Huguenot immigration?

**Ground Truth**: `Lutheran and Reformed | Lutheran and Reformed | Lutheran and Reformed`

**Context Snippet**:

> Around 1685, Huguenot refugees found a safe haven in the Lutheran and Reformed states in Germany and Scandinavia. Nearly 50,000 Huguenots established themselves in Germany, 20,000 of whom were welcomed in Brandenburg-Prussia, where they were granted special privileges (Edict of Potsdam) and churches...

**Answers**:


| Method                         | Answer                | F1    | EM  |
| ------------------------------ | --------------------- | ----- | --- |
| Baseline-Standard (No Privacy) | Lutheran and Reformed | 1.000 | 1   |
| Baseline-Sanitized (DP-ICL)    | Lutheran and Reformed | 1.000 | 1   |
| **SDP-ICL (Proposed)**         | lutheran and reformed | 1.000 | 1   |


**SDP Ensemble Top Answers** (N=100 total): "lutheran and reformed" × 100

---

### Case 3 (SUCCESS)

**自動判斷原因**：SDP-ICL F1=1.000 (EM=1); 優於或持平 Baseline-Standard (F1=1.000); 優於 Baseline-Sanitized (F1=1.000)


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `57263cfcec44d21400f3dc8d` |
| Model             | `qwen`                     |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.5189 sec/query`         |
| Answer Diversity  | `0.01`                     |


**Question**: What was Apple Talk 

**Ground Truth**: `a proprietary suite of networking protocols developed by Apple Inc. in 1985 | a proprietary suite of networking protocols developed by Apple Inc | proprietary suite of networking protocols`

**Context Snippet**:

> AppleTalk was a proprietary suite of networking protocols developed by Apple Inc. in 1985 for Apple Macintosh computers. It was the primary protocol used by Apple devices through the 1980s and 90s. AppleTalk included features that allowed local area networks to be established ad hoc without the requ...

**Answers**:


| Method                         | Answer                                      | F1    | EM  |
| ------------------------------ | ------------------------------------------- | ----- | --- |
| Baseline-Standard (No Privacy) | a proprietary suite of networking protocols | 1.000 | 1   |
| Baseline-Sanitized (DP-ICL)    | a proprietary suite of networking protocols | 1.000 | 1   |
| **SDP-ICL (Proposed)**         | proprietary suite of networking protocols   | 1.000 | 1   |


**SDP Ensemble Top Answers** (N=100 total): "proprietary suite of networking protocols" × 100

---

### Case 4 (SUCCESS)

**自動判斷原因**：SDP-ICL F1=1.000 (EM=1)


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `57265c10f1498d1400e8dd37` |
| Model             | `qwen`                     |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.8244 sec/query`         |
| Answer Diversity  | `0.02`                     |


**Question**: What happens when bathocyroe and ocyropsis clap their lobes together?

**Ground Truth**: `jet of expelled water drives them backwards very quickly. | jet of expelled water drives them backwards very quickly | expelled water drives them backwards very quickly`

**Context Snippet**:

> Lobates have eight comb-rows, originating at the aboral pole and usually not extending beyond the body to the lobes; in species with (four) auricles, the cilia edging the auricles are extensions of cilia in four of the comb rows. Most lobates are quite passive when moving through the water, using th...

**Answers**:


| Method                         | Answer                                                   | F1    | EM  |
| ------------------------------ | -------------------------------------------------------- | ----- | --- |
| Baseline-Standard (No Privacy) | escape from danger                                       | 0.000 | 0   |
| Baseline-Sanitized (DP-ICL)    | escape from danger                                       | 0.000 | 0   |
| **SDP-ICL (Proposed)**         | jet of expelled water drives them backwards very quickly | 1.000 | 1   |


**SDP Ensemble Top Answers** (N=100 total): "jet of expelled water drives them backwards very quickly" × 92 | "escape from danger" × 8

---

### Case 5 (SUCCESS)

**自動判斷原因**：SDP-ICL F1=1.000 (EM=1); 優於或持平 Baseline-Standard (F1=1.000); 優於 Baseline-Sanitized (F1=1.000)


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `571c3a685efbb31900334db3` |
| Model             | `qwen`                     |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.631 sec/query`          |
| Answer Diversity  | `0.02`                     |


**Question**: Of what group in the periodic table is oxygen a member?

**Ground Truth**: `chalcogen | chalcogen | chalcogen | chalcogen | the chalcogen group`

**Context Snippet**:

> Oxygen is a chemical element with symbol O and atomic number 8. It is a member of the chalcogen group on the periodic table and is a highly reactive nonmetal and oxidizing agent that readily forms compounds (notably oxides) with most elements. By mass, oxygen is the third-most abundant element in th...

**Answers**:


| Method                         | Answer          | F1    | EM  |
| ------------------------------ | --------------- | ----- | --- |
| Baseline-Standard (No Privacy) | chalcogen group | 1.000 | 1   |
| Baseline-Sanitized (DP-ICL)    | chalcogen       | 1.000 | 1   |
| **SDP-ICL (Proposed)**         | chalcogen group | 1.000 | 1   |


**SDP Ensemble Top Answers** (N=100 total): "chalcogen group" × 88 | "chalcogen" × 12

---

## 2. Partial Success Cases（部分成功案例，0.4 ≤ F1 < 0.8）

### Case 1 (PARTIAL)

**自動判斷原因**：SDP-ICL F1=0.625 (介於 0.4~0.8); EM=0 但語意可能部分正確，適合說明 metric gap


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `57287e512ca10214002da3fa` |
| Model             | `qwen`                     |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.7468 sec/query`         |
| Answer Diversity  | `0.04`                     |


**Question**: How well did the Mongol Emperors know Chinese?

**Ground Truth**: `could not master written Chinese, but they could generally converse well | could not master written Chinese, but they could generally converse well | well`

**Context Snippet**:

> Since its invention in 1269, the 'Phags-pa script, a unified script for spelling Mongolian, Tibetan, and Chinese languages, was preserved in the court until the end of the dynasty. Most of the Emperors could not master written Chinese, but they could generally converse well in the language. The Mong...

**Answers**:


| Method                         | Answer                           | F1    | EM  |
| ------------------------------ | -------------------------------- | ----- | --- |
| Baseline-Standard (No Privacy) | could not master written Chinese | 0.625 | 0   |
| Baseline-Sanitized (DP-ICL)    | could not master written Chinese | 0.625 | 0   |
| **SDP-ICL (Proposed)**         | could not master written chinese | 0.625 | 0   |


**SDP Ensemble Top Answers** (N=100 total): "could not master written chinese" × 70 | "could generally converse well in language" × 27 | "could generally converse well" × 2 | "could not master written chinese but could generally converse well in language" × 1

---

### Case 2 (PARTIAL)

**自動判斷原因**：SDP-ICL F1=0.571 (介於 0.4~0.8); EM=0 但語意可能部分正確，適合說明 metric gap


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `57274b35f1498d1400e8f5d6` |
| Model             | `qwen`                     |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.6078 sec/query`         |
| Answer Diversity  | `0.01`                     |


**Question**: How are ergänzungsschulen funded?

**Ground Truth**: `tuition | tuition | tuition fees`

**Context Snippet**:

> Ergänzungsschulen are secondary or post-secondary (non-tertiary) schools, which are run by private individuals, private organizations or rarely, religious groups and offer a type of education which is not available at public schools. Most of these schools are vocational schools. However, these vocat...

**Answers**:


| Method                         | Answer                               | F1    | EM  |
| ------------------------------ | ------------------------------------ | ----- | --- |
| Baseline-Standard (No Privacy) | charging their students tuition fees | 0.571 | 0   |
| Baseline-Sanitized (DP-ICL)    | charging their students tuition fees | 0.571 | 0   |
| **SDP-ICL (Proposed)**         | charging their students tuition fees | 0.571 | 0   |


**SDP Ensemble Top Answers** (N=100 total): "charging their students tuition fees" × 100

---

### Case 3 (PARTIAL)

**自動判斷原因**：SDP-ICL F1=0.571 (介於 0.4~0.8); EM=0 但語意可能部分正確，適合說明 metric gap


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `5711119cb654c5140001fae7` |
| Model             | `qwen`                     |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.5905 sec/query`         |
| Answer Diversity  | `0.04`                     |


**Question**: What made emigration to these colonies attractive?

**Ground Truth**: `they were accepted and allowed to worship freely | allowed to worship freely | they were accepted and allowed to worship freely`

**Context Snippet**:

> The bulk of Huguenot émigrés relocated to Protestant European nations such as England, Wales, Scotland, Denmark, Sweden, Switzerland, the Dutch Republic, the Electorate of Brandenburg and Electorate of the Palatinate in the Holy Roman Empire, the Duchy of Prussia, the Channel Islands, and Ireland. T...

**Answers**:


| Method                         | Answer             | F1    | EM  |
| ------------------------------ | ------------------ | ----- | --- |
| Baseline-Standard (No Privacy) | freedom to worship | 0.571 | 0   |
| Baseline-Sanitized (DP-ICL)    | freedom to worship | 0.571 | 0   |
| **SDP-ICL (Proposed)**         | freedom to worship | 0.571 | 0   |


**SDP Ensemble Top Answers** (N=100 total): "freedom to worship" × 90 | "freely to worship" × 4 | "they were accepted and allowed to worship freely" × 4 | "freely worship" × 2

---

### Case 4 (PARTIAL)

**自動判斷原因**：SDP-ICL F1=0.571 (介於 0.4~0.8); EM=0 但語意可能部分正確，適合說明 metric gap


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `57273a465951b619008f8701` |
| Model             | `qwen`                     |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.515 sec/query`          |
| Answer Diversity  | `0.03`                     |


**Question**: What percentile of gross domestic product is construction comprised of?

**Ground Truth**: `six to nine percent | six to nine percent | six to nine percent`

**Context Snippet**:

> Construction is the process of constructing a building or infrastructure. Construction differs from manufacturing in that manufacturing typically involves mass production of similar items without a designated purchaser, while construction typically takes place on location for a known client. Constru...

**Answers**:


| Method                         | Answer             | F1    | EM  |
| ------------------------------ | ------------------ | ----- | --- |
| Baseline-Standard (No Privacy) | 6-9%               | 0.000 | 0   |
| Baseline-Sanitized (DP-ICL)    | six to ninepercent | 0.571 | 0   |
| **SDP-ICL (Proposed)**         | six to ninepercent | 0.571 | 0   |


**SDP Ensemble Top Answers** (N=100 total): "six to ninepercent" × 69 | "69" × 20 | "six to nine percent" × 11

---

### Case 5 (PARTIAL)

**自動判斷原因**：SDP-ICL F1=0.667 (介於 0.4~0.8); EM=0 但語意可能部分正確，適合說明 metric gap


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `57265c10f1498d1400e8dd36` |
| Model             | `qwen`                     |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.7939 sec/query`         |
| Answer Diversity  | `0.01`                     |


**Question**: What does the bathocyroe and ocyropsis do to escape danger?

**Ground Truth**: `by clapping their lobes | clapping their lobes | clapping their lobes`

**Context Snippet**:

> Lobates have eight comb-rows, originating at the aboral pole and usually not extending beyond the body to the lobes; in species with (four) auricles, the cilia edging the auricles are extensions of cilia in four of the comb rows. Most lobates are quite passive when moving through the water, using th...

**Answers**:


| Method                         | Answer           | F1    | EM  |
| ------------------------------ | ---------------- | ----- | --- |
| Baseline-Standard (No Privacy) | clap their lobes | 0.667 | 0   |
| Baseline-Sanitized (DP-ICL)    | clap their lobes | 0.667 | 0   |
| **SDP-ICL (Proposed)**         | clap their lobes | 0.667 | 0   |


**SDP Ensemble Top Answers** (N=100 total): "clap their lobes" × 100

---

## 3. Failure Cases（失敗案例，F1 ≤ 0.2）

### Case 1 (FAILURE)

**自動判斷原因**：SDP-ICL F1=0.000 (<=  0.2); ensemble concentrated (diversity=0.050) but still wrong, suggests first-stage error; Baseline-Standard answered correctly (F1=1.000), SDP introduced error -- good privacy-utility tradeoff example


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `5727e8424b864d1900163fc0` |
| Model             | `qwen`                     |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.5095 sec/query`         |
| Answer Diversity  | `0.05`                     |


**Question**: How many Americans are richer than more than half of all citizens?

**Ground Truth**: `400 | 400 | 400`

**Context Snippet**:

> According to PolitiFact the top 400 richest Americans "have more wealth than half of all Americans combined." According to the New York Times on July 22, 2014, the "richest 1 percent in the United States now own more wealth than the bottom 90 percent". Inherited wealth may help explain why many Amer...

**Answers**:


| Method                         | Answer | F1    | EM  |
| ------------------------------ | ------ | ----- | --- |
| Baseline-Standard (No Privacy) | 400    | 1.000 | 1   |
| Baseline-Sanitized (DP-ICL)    | 400    | 1.000 | 1   |
| **SDP-ICL (Proposed)**         | half   | 0.000 | 0   |


**SDP Ensemble Top Answers** (N=100 total): "half" × 69 | "400" × 14 | "half of all americans" × 9 | "more than half" × 6 | "less than half" × 2

---

### Case 2 (FAILURE)

**自動判斷原因**：SDP-ICL F1=0.182 (<=  0.2); ensemble concentrated (diversity=0.060) but still wrong, suggests first-stage error; Baseline-Standard answered correctly (F1=1.000), SDP introduced error -- good privacy-utility tradeoff example; Both SDP and Baseline-Sanitized fail while Standard succeeds, suggests sanitization itself causes the accuracy drop


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `57300a25a23a5019007fcc98` |
| Model             | `qwen`                     |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.7831 sec/query`         |
| Answer Diversity  | `0.06`                     |


**Question**: What was a long term goal of French foreign policy along the Rhine?

**Ground Truth**: `Establishing "natural borders" | Establishing "natural borders" | Establishing "natural borders" on the Rhine`

**Context Snippet**:

> Since the Peace of Westphalia, the Upper Rhine formed a contentious border between France and Germany. Establishing "natural borders" on the Rhine was a long-term goal of French foreign policy, since the Middle Ages, though the language border was – and is – far more to the west. French leaders, suc...

**Answers**:


| Method                         | Answer                                      | F1    | EM  |
| ------------------------------ | ------------------------------------------- | ----- | --- |
| Baseline-Standard (No Privacy) | establishing "natural borders" on the Rhine | 1.000 | 1   |
| Baseline-Sanitized (DP-ICL)    | to annex lands west of the Rhine            | 0.182 | 0   |
| **SDP-ICL (Proposed)**         | to annex lands west of rhine                | 0.182 | 0   |


**SDP Ensemble Top Answers** (N=100 total): "to annex lands west of rhine" × 40 | "natural borders" × 29 | "establishing natural borders on rhine" × 25 | "natural borders on rhine" × 4 | "establishing natural borders" × 1 | ... 另有 1 種不同答案

---

### Case 3 (FAILURE)

**自動判斷原因**：SDP-ICL F1=0.000 (<=  0.2)


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `571114cfb654c5140001fb0a` |
| Model             | `qwen`                     |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `2.2617 sec/query`         |
| Answer Diversity  | `0.1`                      |


**Question**: How did the new king react to the Huguenots?

**Ground Truth**: `acted increasingly aggressively to force the Huguenots to convert | aggressively | increasingly aggressively`

**Context Snippet**:

> Louis XIV gained the throne in 1643 and acted increasingly aggressively to force the Huguenots to convert. At first he sent missionaries, backed by a fund to financially reward converts to Catholicism. Then he imposed penalties, closed Huguenot schools and excluded them from favored professions. Esc...

**Answers**:


| Method                         | Answer                                                                                                                                 | F1    | EM  |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- | ----- | --- |
| Baseline-Standard (No Privacy) | sent missionaries, imposed penalties, closed schools, excluded from professions, instituted dragonnades, issued Edict of Fontainebleau | 0.000 | 0   |
| Baseline-Sanitized (DP-ICL)    | forced conversion                                                                                                                      | 0.000 | 0   |
| **SDP-ICL (Proposed)**         | sent missionaries imposed penalties closed schools excluded from professions instituted dragonnades issued edict of fontainebleau      | 0.000 | 0   |


**SDP Ensemble Top Answers** (N=100 total): "sent missionaries imposed penalties closed schools excluded from professions instituted dragonnades issued edict of fontainebleau" × 60 | "sent missionaries imposed penalties closed huguenot schools and excluded them from favored professions instituted dragonnades issued edict of fontainebleau" × 18 | "forced huguenots to convert" × 6 | "forced them to convert" × 6 | "increasingly aggressively" × 3 | ... 另有 5 種不同答案

---

### Case 4 (FAILURE)

**自動判斷原因**：SDP-ICL F1=0.000 (<=  0.2); ensemble concentrated (diversity=0.060) but still wrong, suggests first-stage error


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `5727e9523acd2414000def98` |
| Model             | `qwen`                     |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.6103 sec/query`         |
| Answer Diversity  | `0.06`                     |


**Question**: What is the term that describes the difference between what higher paid and lower paid professionals earn?

**Ground Truth**: `productivity gap | productivity gap | productivity gap`

**Context Snippet**:

> Neoclassical economics views inequalities in the distribution of income as arising from differences in value added by labor, capital and land. Within labor income distribution is due to differences in value added by different classifications of workers. In this perspective, wages and profits are det...

**Answers**:


| Method                         | Answer                     | F1    | EM  |
| ------------------------------ | -------------------------- | ----- | --- |
| Baseline-Standard (No Privacy) | differences in value added | 0.000 | 0   |
| Baseline-Sanitized (DP-ICL)    | differences in value added | 0.000 | 0   |
| **SDP-ICL (Proposed)**         | differences in value added | 0.000 | 0   |


**SDP Ensemble Top Answers** (N=100 total): "differences in value added" × 86 | "inequalities in distribution of income" × 5 | "productivity gap" × 4 | "inequalities" × 3 | "inequality" × 1 | ... 另有 1 種不同答案

---

### Case 5 (FAILURE)

**自動判斷原因**：SDP-ICL F1=0.095 (<=  0.2); ensemble concentrated (diversity=0.050) but still wrong, suggests first-stage error


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `5733d2dbd058e614000b633b` |
| Model             | `qwen`                     |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.8592 sec/query`         |
| Answer Diversity  | `0.05`                     |


**Question**: What was the significance of British win?

**Ground Truth**: `confirming Britain's position as the dominant colonial power in eastern North America | confirming Britain's position as the dominant colonial power in eastern North America | confirming Britain's position as the dominant colonial power in eastern North America | dominant colonial power | confirming Britain's position as the dominant colonial power in eastern North America`

**Context Snippet**:

> The outcome was one of the most significant developments in a century of Anglo-French conflict. France ceded its territory east of the Mississippi to Great Britain. It ceded French Louisiana west of the Mississippi River (including New Orleans) to its ally Spain, in compensation for Spain's loss to ...

**Answers**:


| Method                         | Answer                                                                         | F1    | EM  |
| ------------------------------ | ------------------------------------------------------------------------------ | ----- | --- |
| Baseline-Standard (No Privacy) | one of the most significant developments in a century of Anglo-French conflict | 0.095 | 0   |
| Baseline-Sanitized (DP-ICL)    | one of the most significant developments in a century of Anglo-French conflict | 0.095 | 0   |
| **SDP-ICL (Proposed)**         | one of most significant developments in century of anglofrench conflict        | 0.095 | 0   |


**SDP Ensemble Top Answers** (N=100 total): "one of most significant developments in century of anglofrench conflict" × 77 | "dominant colonial power in eastern north america" × 10 | "france ceded its territory east of mississippi to great britain" × 9 | "anglofrench conflict" × 3 | "most significant developments in century of anglofrench conflict" × 1

---

