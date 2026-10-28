# Sanad evaluation report

- dataset: /home/muaaz/Desktop/N-Job/portfolio/sanad/eval/questions.jsonl (50 questions; dev 25, test 25)
- corpus snapshot: 10 documents, 585 chunks
- embedding model: `ollama:bge-m3`

## Retrieval

| mode | split | k | n | recall@k | MRR | nDCG@k | evidence coverage |
|---|---|---|---|---|---|---|---|
| hybrid | test | 5 | 20 | 0.950 | 0.792 | 0.833 | 0.950 |
| lexical | test | 5 | 20 | 0.900 | 0.664 | 0.722 | 0.900 |

## End-to-end answers

- model: `ollama:qwen2.5:7b`; split: test; top-k: 5
- answer accuracy (gold answer present): 0.950 (20 answerable)
- false abstention on answerable: 0.050
- fully grounded answers: 0.950
- abstention on unanswerable: 1.000 (5 unanswerable)
- citation validity among answered: 1.000
- mean grounded-sentence ratio: 1.000
- latency p50/p95: 8142 / 9098 ms

### Per question

| id | answerable | status | correct | grounded | answer |
|---|---|---|---|---|---|
| q02 | True | answered | True | 1.0 | تقع مدينة الرياض على ارتفاع 600 متر فوق مستوى سطح البحر [S1, S2, S3]. |
| q03 | True | answered | True | 1.0 | وفقًا للسندات المقدمة [S1] و[S2]، تبلغ مساحة مدينة الرياض في الوقت الحاضر 3,115 كم². |
| q06 | True | answered | True | 1.0 | إجمالي طول شبكة مترو الرياض يبلغ 176 كيلومتراً [S1, S2, S3, S4]. |
| q08 | True | answered | True | 1.0 | بلغت تكلفة مشروع مترو الرياض 22.5 مليار دولار [S1, S3]. |
| q10 | True | answered | True | 1.0 | تحاول كاليفورنيا إنشاء 17 محطة لتحلية المياه على المستوى الصناعي [S1]. |
| q12 | True | answered | True | 1.0 | سعة المقطرات في المحطة التي أُنشئت في سواكن كانت 350 طن/يوم كما ذكر في المصدر [S2]. |
| q14 | True | answered | True | 1.0 | مقر جامعة الجزيرة يقع في مدينة ود مدني حاضرة ولاية الجزيرة [S1]. |
| q15 | True | answered | True | 1.0 | أشهر كليات جامعة الجزيرة هي كلية الطب [S1, S2]. |
| q17 | True | answered | True | 1.0 | في 30 نوفمبر 2022 أطلقت أوبن إيه آي روبوت المحادثة «تشات جي بي تي» [S1]. |
| q19 | True | answered | True | 1.0 | نظام ألفا فولد الذي نال عنه مطوروه نصف جائزة نوبل في الكيمياء 2024 [S1]. |
| q21 | True | answered | True | 1.0 | يبلغ متوسط الأمطار السنوية في الخرطوم حوالي 155 مليمتر (6.1 بوصة) سنوياً [S2]. |
| q23 | True | insufficient | False | None | INSUFFICIENT_EVIDENCE |
| q26 | True | answered | True | 1.0 | The Riyadh Metro was inaugurated on 27 November 2024, with the opening of Lines 1, 4, and 6 on 1 December 2024, followed by Lines 2 and 5 on 15 December 2024, and Line 3 on 5 Janua |
| q27 | True | answered | True | 1.0 | The Riyadh Metro was recognized by Guinness World Records as the world's longest fully automated, driverless metro network, spanning 176 kilometres (109 mi) with six lines and 85 s |
| q29 | True | answered | True | 1.0 | According to a 2019 review in Science of the Total Environment, around 95 million cubic meters per day of desalinated water is produced worldwide [S1]. |
| q31 | True | answered | True | 1.0 | The global energy intensity of desalination in 2018 was about 3 kWh/m3 [S1, S2]. |
| q33 | True | answered | True | 1.0 | Khartoum was founded in 1821 by Muhammad Ali Pasha [S1, S3, S4]. |
| q35 | True | answered | True | 1.0 | About 50% of drinking water in Saudi Arabia comes from desalination [S1, S2]. |
| q37 | True | answered | True | 1.0 | 40% of the country's drinking water comes from the mining of non-renewable groundwater [S3]. |
| q40 | True | answered | True | 1.0 | The Riyadh Metro holds the distinction of being the world's longest fully automated, driverless metro network, spanning 176 kilometres (109 mi) with six lines and 85 stations [S1,  |
| q46 | False | insufficient | True | None | INSUFFICIENT_EVIDENCE |
| q47 | False | insufficient | True | None | INSUFFICIENT_EVIDENCE |
| q48 | False | insufficient | True | None | INSUFFICIENT_EVIDENCE |
| q49 | False | insufficient | True | None | INSUFFICIENT_EVIDENCE |
| q50 | False | insufficient | True | None | INSUFFICIENT_EVIDENCE |

## Method notes

- Retrieval hit = a retrieved chunk contains the labeled evidence quote verbatim after Arabic/Latin normalization; labels are validated against the corpus.
- Answer accuracy uses normalized gold-answer containment; it is strict about the fact and lenient about surrounding wording.
- "Unsupported" = answered without a grounded, valid citation.

