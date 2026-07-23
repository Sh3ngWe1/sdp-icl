# SDP-ICL Qualitative Candidates Report

> 本報告由 `scripts/find_qualitative_cases.py` 自動產生。
> 所有案例均來自真實實驗結果，請人工審查後決定是否放入論文。

## 1. Success Cases（成功案例，F1 ≥ 0.8 或 EM = 1）

### Case 1 (SUCCESS)

**自動判斷原因**：SDP-ICL F1=1.000 (EM=1); 優於或持平 Baseline-Standard (F1=0.706); 優於 Baseline-Sanitized (F1=0.706)


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `5730a4d02461fd1900a9cf2b` |
| Model             | `llama`                    |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.9426 sec/query`         |
| Answer Diversity  | `0.03`                     |


**Question**: Where did France focus its efforts to rebuild its empire?

**Ground Truth**: `Africa | North and West Africa | North and West Africa | North and West Africa, as well as South-East Asia, with other conquests in Central and East Africa, as well as the South Pacific | North and West Africa, as well as South-East Asia,`

**Context Snippet**:

> France took control of Algeria in 1830 but began in earnest to rebuild its worldwide empire after 1850, concentrating chiefly in North and West Africa, as well as South-East Asia, with other conquests in Central and East Africa, as well as the South Pacific. Republicans, at first hostile to empire, ...

**Answers**:


| Method                         | Answer                                          | F1    | EM  |
| ------------------------------ | ----------------------------------------------- | ----- | --- |
| Baseline-Standard (No Privacy) | North and West Africa,                          |       |     |
| South-East Asia,               |                                                 |       |     |
| Central and East Africa,       |                                                 |       |     |
| the South Pacific              | 0.706                                           | 0     |     |
| Baseline-Sanitized (DP-ICL)    | North and West Africa,                          |       |     |
| South-East Asia,               |                                                 |       |     |
| Central and East Africa,       |                                                 |       |     |
| the South Pacific              | 0.706                                           | 0     |     |
| **SDP-ICL (Proposed)**         | north and west africa as well as southeast asia | 1.000 | 1   |


**SDP Ensemble Top Answers** (N=100 total): "north and west africa as well as southeast asia" × 96 | "north and west africa southeast asia" × 3 | "north africa southeast asia central africa east africa south pacific north and west africa" × 1

---

### Case 2 (SUCCESS)

**自動判斷原因**：SDP-ICL F1=1.000 (EM=1); 優於或持平 Baseline-Standard (F1=1.000); 優於 Baseline-Sanitized (F1=0.800)


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `57111ab8a58dae1900cd6c3c` |
| Model             | `llama`                    |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `2.0669 sec/query`         |
| Answer Diversity  | `0.01`                     |


**Question**: What protestant religions made Northern European counties safe for Huguenot immigration?

**Ground Truth**: `Lutheran and Reformed | Lutheran and Reformed | Lutheran and Reformed`

**Context Snippet**:

> Around 1685, Huguenot refugees found a safe haven in the Lutheran and Reformed states in Germany and Scandinavia. Nearly 50,000 Huguenots established themselves in Germany, 20,000 of whom were welcomed in Brandenburg-Prussia, where they were granted special privileges (Edict of Potsdam) and churches...

**Answers**:


| Method                         | Answer                | F1    | EM  |
| ------------------------------ | --------------------- | ----- | --- |
| Baseline-Standard (No Privacy) | Lutheran and Reformed | 1.000 | 1   |
| Baseline-Sanitized (DP-ICL)    | Lutheran              |       |     |
| Reformed                       | 0.800                 | 0     |     |
| **SDP-ICL (Proposed)**         | lutheran and reformed | 1.000 | 1   |


**SDP Ensemble Top Answers** (N=100 total): "lutheran and reformed" × 100

---

### Case 3 (SUCCESS)

**自動判斷原因**：SDP-ICL F1=1.000 (EM=1); 優於或持平 Baseline-Standard (F1=1.000); 優於 Baseline-Sanitized (F1=1.000)


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `57263cfcec44d21400f3dc8d` |
| Model             | `llama`                    |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.7041 sec/query`         |
| Answer Diversity  | `0.04`                     |


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


**SDP Ensemble Top Answers** (N=100 total): "proprietary suite of networking protocols" × 70 | "networking protocols" × 15 | "appletalk" × 9 | "suite of networking protocols" × 6

---

### Case 4 (SUCCESS)

**自動判斷原因**：SDP-ICL F1=1.000 (EM=1); 優於或持平 Baseline-Standard (F1=1.000); 優於 Baseline-Sanitized (F1=1.000)


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `5729f0db6aef051400155126` |
| Model             | `llama`                    |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.717 sec/query`          |
| Answer Diversity  | `0.01`                     |


**Question**: The Amazon region is home to how many species of insect?

**Ground Truth**: `2.5 million | about 2.5 million | 2.5 million`

**Context Snippet**:

> The region is home to about 2.5 million insect species, tens of thousands of plants, and some 2,000 birds and mammals. To date, at least 40,000 plant species, 2,200 fishes, 1,294 birds, 427 mammals, 428 amphibians, and 378 reptiles have been scientifically classified in the region. One in five of al...

**Answers**:


| Method                         | Answer      | F1    | EM  |
| ------------------------------ | ----------- | ----- | --- |
| Baseline-Standard (No Privacy) | 2.5 million | 1.000 | 1   |
| Baseline-Sanitized (DP-ICL)    | 2.5 million | 1.000 | 1   |
| **SDP-ICL (Proposed)**         | 25 million  | 1.000 | 1   |


**SDP Ensemble Top Answers** (N=100 total): "25 million" × 100

---

### Case 5 (SUCCESS)

**自動判斷原因**：SDP-ICL F1=1.000 (EM=1); 優於或持平 Baseline-Standard (F1=1.000); 優於 Baseline-Sanitized (F1=1.000)


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `571c3a685efbb31900334db3` |
| Model             | `llama`                    |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.8838 sec/query`         |
| Answer Diversity  | `0.01`                     |


**Question**: Of what group in the periodic table is oxygen a member?

**Ground Truth**: `chalcogen | chalcogen | chalcogen | chalcogen | the chalcogen group`

**Context Snippet**:

> Oxygen is a chemical element with symbol O and atomic number 8. It is a member of the chalcogen group on the periodic table and is a highly reactive nonmetal and oxidizing agent that readily forms compounds (notably oxides) with most elements. By mass, oxygen is the third-most abundant element in th...

**Answers**:


| Method                         | Answer    | F1    | EM  |
| ------------------------------ | --------- | ----- | --- |
| Baseline-Standard (No Privacy) | chalcogen | 1.000 | 1   |
| Baseline-Sanitized (DP-ICL)    | chalcogen | 1.000 | 1   |
| **SDP-ICL (Proposed)**         | chalcogen | 1.000 | 1   |


**SDP Ensemble Top Answers** (N=100 total): "chalcogen" × 100

---

## 2. Partial Success Cases（部分成功案例，0.4 ≤ F1 < 0.8）

### Case 1 (PARTIAL)

**自動判斷原因**：SDP-ICL F1=0.600 (介於 0.4~0.8); EM=0 但語意可能部分正確，適合說明 metric gap


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `5725cc38ec44d21400f3d5bb` |
| Model             | `llama`                    |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `2.0387 sec/query`         |
| Answer Diversity  | `0.04`                     |


**Question**: How did the principle treaties that form the European Union begin?

**Ground Truth**: `with common rules for coal and steel, and then atomic energy | with common rules for coal and steel | with common rules for coal and steel | with common rules for coal and steel`

**Context Snippet**:

> The principal Treaties that form the European Union began with common rules for coal and steel, and then atomic energy, but more complete and formal institutions were established through the Treaty of Rome 1957 and the Maastricht Treaty 1992 (now: TFEU). Minor amendments were made during the 1960s a...

**Answers**:


| Method                         | Answer         | F1    | EM  |
| ------------------------------ | -------------- | ----- | --- |
| Baseline-Standard (No Privacy) | coal and steel | 0.600 | 0   |
| Baseline-Sanitized (DP-ICL)    | coal and steel | 0.600 | 0   |
| **SDP-ICL (Proposed)**         | coal and steel | 0.600 | 0   |


**SDP Ensemble Top Answers** (N=100 total): "coal and steel" × 91 | "common rules for coal and steel" × 5 | "with common rules for coal and steel" × 3 | "common rules for coal and steel atomic energy" × 1

---

### Case 2 (PARTIAL)

**自動判斷原因**：SDP-ICL F1=0.600 (介於 0.4~0.8); EM=0 但語意可能部分正確，適合說明 metric gap


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `572941273f37b319004781b1` |
| Model             | `llama`                    |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.8076 sec/query`         |
| Answer Diversity  | `0.06`                     |


**Question**: What was the verdict on other alleged errors?

**Ground Truth**: `"generally unfounded and also marginal to the assessment" | generally unfounded and also marginal to the assessment | generally unfounded and also marginal to the assessment`

**Context Snippet**:

> Former IPCC chairman Robert Watson has said "The mistakes all appear to have gone in the direction of making it seem like climate change is more serious by overstating the impact. That is worrying. The IPCC needs to look at this trend in the errors and ask why it happened". Martin Parry, a climate e...

**Answers**:


| Method                         | Answer                 | F1    | EM    |
| ------------------------------ | ---------------------- | ----- | ----- |
| Baseline-Standard (No Privacy) | *N/A*                  | *N/A* | *N/A* |
| Baseline-Sanitized (DP-ICL)    | *N/A*                  | *N/A* | *N/A* |
| **SDP-ICL (Proposed)**         | unfounded and marginal | 0.600 | 0     |


**SDP Ensemble Top Answers** (N=100 total): "unfounded and marginal" × 77 | "generally unfounded and also marginal" × 13 | "unfounded" × 5 | "generally unfounded" × 2 | "generally unfounded and also marginal to assessment" × 2 | ... 另有 1 種不同答案

---

### Case 3 (PARTIAL)

**自動判斷原因**：SDP-ICL F1=0.600 (介於 0.4~0.8); EM=0 但語意可能部分正確，適合說明 metric gap


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `5728eb1a3acd2414000e01c9` |
| Model             | `llama`                    |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `2.2774 sec/query`         |
| Answer Diversity  | `0.17`                     |


**Question**: Anarchists do not want to accept punishment for what reason?

**Ground Truth**: `does not infringe the rights of others | don't believe in the legitimacy of any government | a violation of criminal law that does not infringe the rights of others | see no need to accept punishment for a violation of criminal law that does not infringe the rights of others | a violation of criminal law that does not infringe the rights of others.`

**Context Snippet**:

> Some civil disobedients feel it is incumbent upon them to accept punishment because of their belief in the validity of the social contract, which is held to bind all to obey the laws that a government meeting certain standards of legitimacy has established, or else suffer the penalties set out in th...

**Answers**:


| Method                         | Answer                      | F1    | EM    |
| ------------------------------ | --------------------------- | ----- | ----- |
| Baseline-Standard (No Privacy) | *N/A*                       | *N/A* | *N/A* |
| Baseline-Sanitized (DP-ICL)    | *N/A*                       | *N/A* | *N/A* |
| **SDP-ICL (Proposed)**         | infringing rights of others | 0.600 | 0     |


**SDP Ensemble Top Answers** (N=100 total): "infringing rights of others" × 31 | "they see no need to accept punishment for violation of criminal law that does not infringe rights of others" × 26 | "because it doesnt infringe rights of others" × 12 | "they dont believe in legitimacy of any government" × 10 | "because it does not infringe rights of others" × 4 | ... 另有 12 種不同答案

---

### Case 4 (PARTIAL)

**自動判斷原因**：SDP-ICL F1=0.600 (介於 0.4~0.8); EM=0 但語意可能部分正確，適合說明 metric gap


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `572a03a06aef0514001551ae` |
| Model             | `llama`                    |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.8077 sec/query`         |
| Answer Diversity  | `0.02`                     |


**Question**: What are the main threats facing the Amazon rainforest in the current century?

**Ground Truth**: `climate change in addition to deforestation | climate change in addition to deforestation | climate change in addition to deforestation`

**Context Snippet**:

> One computer model of future climate change caused by greenhouse gas emissions shows that the Amazon rainforest could become unsustainable under conditions of severely reduced rainfall and increased temperatures, leading to an almost complete loss of rainforest cover in the basin by 2100. However, s...

**Answers**:


| Method                         | Answer                           | F1    | EM    |
| ------------------------------ | -------------------------------- | ----- | ----- |
| Baseline-Standard (No Privacy) | *N/A*                            | *N/A* | *N/A* |
| Baseline-Sanitized (DP-ICL)    | *N/A*                            | *N/A* | *N/A* |
| **SDP-ICL (Proposed)**         | climate change and deforestation | 0.600 | 0     |


**SDP Ensemble Top Answers** (N=100 total): "climate change and deforestation" × 80 | "deforestation and climate change" × 20

---

### Case 5 (PARTIAL)

**自動判斷原因**：SDP-ICL F1=0.600 (介於 0.4~0.8); EM=0 但語意可能部分正確，適合說明 metric gap


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `57294baaaf94a219006aa26b` |
| Model             | `llama`                    |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.8557 sec/query`         |
| Answer Diversity  | `0.01`                     |


**Question**: What role did Michael Oppenheimer have in the IPCC's reports?

**Ground Truth**: `coordinating lead author of the Fifth Assessment Report | participant in the IPCC and coordinating lead author of the Fifth Assessment Report | coordinating lead author of the Fifth Assessment Report`

**Context Snippet**:

> Michael Oppenheimer, a long-time participant in the IPCC and coordinating lead author of the Fifth Assessment Report conceded in Science Magazine's State of the Planet 2008-2009 some limitations of the IPCC consensus approach and asks for concurring, smaller assessments of special problems instead o...

**Answers**:


| Method                         | Answer                   | F1    | EM    |
| ------------------------------ | ------------------------ | ----- | ----- |
| Baseline-Standard (No Privacy) | *N/A*                    | *N/A* | *N/A* |
| Baseline-Sanitized (DP-ICL)    | *N/A*                    | *N/A* | *N/A* |
| **SDP-ICL (Proposed)**         | coordinating lead author | 0.600 | 0     |


**SDP Ensemble Top Answers** (N=100 total): "coordinating lead author" × 100

---

## 3. Failure Cases（失敗案例，F1 ≤ 0.2）

### Case 1 (FAILURE)

**自動判斷原因**：SDP-ICL F1=0.000 (<=  0.2); Baseline-Standard answered correctly (F1=1.000), SDP introduced error -- good privacy-utility tradeoff example


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `571112ada58dae1900cd6bd0` |
| Model             | `llama`                    |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.9256 sec/query`         |
| Answer Diversity  | `0.1`                      |


**Question**: Other theories of the word's origin can be generally classed as what?

**Ground Truth**: `double or triple non-French linguistic origins | non-French linguistic origins`

**Context Snippet**:

> Some disagree with such double or triple non-French linguistic origins, arguing that for the word to have spread into common use in France, it must have originated in the French language. The "Hugues hypothesis" argues that the name was derived by association with Hugues Capet, king of France, who r...

**Answers**:


| Method                         | Answer                        | F1    | EM  |
| ------------------------------ | ----------------------------- | ----- | --- |
| Baseline-Standard (No Privacy) | non-French linguistic origins | 1.000 | 1   |
| Baseline-Sanitized (DP-ICL)    | non-French linguistic origins | 1.000 | 1   |
| **SDP-ICL (Proposed)**         | hugues hypothesis             | 0.000 | 0   |


**SDP Ensemble Top Answers** (N=100 total): "hugues hypothesis" × 39 | "double or triple nonfrench linguistic origins" × 29 | "nonfrench linguistic origins" × 24 | "french language" × 2 | "hugues hypothesis is one theory however since you asked for other theories i will provide more general answer other theories" × 1 | ... 另有 5 種不同答案

---

### Case 2 (FAILURE)

**自動判斷原因**：SDP-ICL F1=0.000 (<=  0.2); ensemble concentrated (diversity=0.090) but still wrong, suggests first-stage error; Baseline-Standard answered correctly (F1=1.000), SDP introduced error -- good privacy-utility tradeoff example


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `572ff932a23a5019007fcbd7` |
| Model             | `llama`                    |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.9218 sec/query`         |
| Answer Diversity  | `0.09`                     |


**Question**: The idea that Islam can be apolitical isn't able to be embraced by whom?

**Ground Truth**: `its supporters | Scholars and observers | Islamism`

**Context Snippet**:

> Islamism is a controversial concept not just because it posits a political role for Islam but also because its supporters believe their views merely reflect Islam, while the contrary idea that Islam is, or can be, apolitical is an error. Scholars and observers who do not believe that Islam is merely...

**Answers**:


| Method                         | Answer                                        | F1    | EM  |
| ------------------------------ | --------------------------------------------- | ----- | --- |
| Baseline-Standard (No Privacy) | Scholars and observers                        | 1.000 | 1   |
| Baseline-Sanitized (DP-ICL)    | Scholars and observers                        | 1.000 | 1   |
| **SDP-ICL (Proposed)**         | muslim intellectuals like javed ahmad ghamidi | 0.000 | 0   |


**SDP Ensemble Top Answers** (N=100 total): "muslim intellectuals like javed ahmad ghamidi" × 77 | "scholars and observers" × 8 | "some scholars and observers" × 5 | "progressive moderates" × 3 | "some" × 2 | ... 另有 4 種不同答案

---

### Case 3 (FAILURE)

**自動判斷原因**：SDP-ICL F1=0.000 (<=  0.2); ensemble concentrated (diversity=0.040) but still wrong, suggests first-stage error; Baseline-Standard answered correctly (F1=1.000), SDP introduced error -- good privacy-utility tradeoff example


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `5725cbb289a1e219009abed4` |
| Model             | `llama`                    |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.8511 sec/query`         |
| Answer Diversity  | `0.04`                     |


**Question**: What was believed to be the cause of devastation to the civilization?

**Ground Truth**: `diseases from Europe | the spread of diseases from Europe | spread of diseases from Europe`

**Context Snippet**:

> The first European to travel the length of the Amazon River was Francisco de Orellana in 1542. The BBC's Unnatural Histories presents evidence that Orellana, rather than exaggerating his claims as previously thought, was correct in his observations that a complex civilization was flourishing along t...

**Answers**:


| Method                         | Answer                          | F1    | EM  |
| ------------------------------ | ------------------------------- | ----- | --- |
| Baseline-Standard (No Privacy) | Spread of diseases from Europe. | 1.000 | 1   |
| Baseline-Sanitized (DP-ICL)    | Spread of diseases from Europe. | 1.000 | 1   |
| **SDP-ICL (Proposed)**         | smallpox                        | 0.000 | 0   |


**SDP Ensemble Top Answers** (N=100 total): "smallpox" × 66 | "diseases from europe" × 18 | "spread of diseases from europe" × 10 | "spread of diseases from europe such as smallpox" × 6

---

### Case 4 (FAILURE)

**自動判斷原因**：SDP-ICL F1=0.000 (<=  0.2); ensemble concentrated (diversity=0.030) but still wrong, suggests first-stage error; Baseline-Standard answered correctly (F1=1.000), SDP introduced error -- good privacy-utility tradeoff example


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `572a020f6aef05140015519b` |
| Model             | `llama`                    |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.9389 sec/query`         |
| Answer Diversity  | `0.03`                     |


**Question**: What amount of the worlds carbon is stored in the Amazon forest?

**Ground Truth**: `10% of the carbon stores | 10% | 10%`

**Context Snippet**:

> Environmentalists are concerned about loss of biodiversity that will result from destruction of the forest, and also about the release of the carbon contained within the vegetation, which could accelerate global warming. Amazonian evergreen forests account for about 10% of the world's terrestrial pr...

**Answers**:


| Method                         | Answer                  | F1    | EM  |
| ------------------------------ | ----------------------- | ----- | --- |
| Baseline-Standard (No Privacy) | 10%                     | 1.000 | 1   |
| Baseline-Sanitized (DP-ICL)    | 10%                     | 1.000 | 1   |
| **SDP-ICL (Proposed)**         | 11 × 1011 metric tonnes | 0.000 | 0   |


**SDP Ensemble Top Answers** (N=100 total): "11 × 1011 metric tonnes" × 73 | "10" × 16 | "11 × 1011 metric tonnes of carbon" × 11

---

### Case 5 (FAILURE)

**自動判斷原因**：SDP-ICL F1=0.000 (<=  0.2); ensemble concentrated (diversity=0.030) but still wrong, suggests first-stage error; Baseline-Standard answered correctly (F1=1.000), SDP introduced error -- good privacy-utility tradeoff example


| 屬性                | 值                          |
| ----------------- | -------------------------- |
| Question ID       | `5729fb003f37b31900478628` |
| Model             | `llama`                    |
| Exemplar Type     | `sanitized`                |
| N (ensemble size) | `100`                      |
| SDP Avg Latency   | `1.8205 sec/query`         |
| Answer Diversity  | `0.03`                     |


**Question**: As a person gets older, what does the skin produce less of?

**Ground Truth**: `vitamin D | vitamin D | vitamin D.`

**Context Snippet**:

> It is conjectured that a progressive decline in hormone levels with age is partially responsible for weakened immune responses in aging individuals. Conversely, some hormones are regulated by the immune system, notably thyroid hormone activity. The age-related decline in immune function is also rela...

**Answers**:


| Method                         | Answer          | F1    | EM  |
| ------------------------------ | --------------- | ----- | --- |
| Baseline-Standard (No Privacy) | vitamin D       | 1.000 | 1   |
| Baseline-Sanitized (DP-ICL)    | vitamin D       | 1.000 | 1   |
| **SDP-ICL (Proposed)**         | cholecalciferol | 0.000 | 0   |


**SDP Ensemble Top Answers** (N=100 total): "cholecalciferol" × 93 | "vitamin d" × 6 | "cholecalciferol via uvb radiation however since you asked about skin itself i will correct my answer to vitamin d" × 1

---

