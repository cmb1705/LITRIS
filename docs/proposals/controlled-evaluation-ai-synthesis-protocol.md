<!-- markdownlint-disable MD013 MD034 -->

# Controlled Evaluation of Batch and Accumulative AI Synthesis for Research-Gap Identification

**Issue**: LITRIS-32r  
**Date**: 2026-04-02  
**Status**: Draft protocol for internal review and preregistration

## Abstract

This proposal describes a controlled follow-on study to an initial five-domain comparison of two AI-assisted literature-synthesis protocols: a batch protocol, in which a model reviews a domain corpus and produces one synthesis at the end of the session, and an accumulative protocol, in which the model updates its synthesis as it reads each source. The pilot suggested that the accumulative protocol produced not only more gap statements but also different ones, including cross-paper tensions, methodological omissions, governance failures, and infrastructure constraints. Those observations are useful, but they do not yet warrant strong inferential claims because the pilot relied on small corpora, abstract-level evidence, same-session comparisons, and unblinded assessment.

The proposed study reframes the problem as a protocol-comparison experiment. It will compare batch, ordered accumulative, and order-shuffled accumulative synthesis across a stratified multi-domain corpus using standardized evidence packets, fresh sessions, replicated runs, fixed output templates, explicit provenance, and blinded expert evaluation. The design is explanatory sequential mixed methods: a confirmatory quantitative analysis will test whether protocol choice affects utility and evidential support, and a qualitative analysis will then characterize the classes of gaps produced by each protocol and explain observed quantitative contrasts. The protocol is structured for preregistration on the Open Science Framework and can be adapted, with limited further specification, into a Stage 1 Registered Report.

## 1. Background and Literature

Research-synthesis workflows have long been targets for automation, but most prior work has focused on search, screening, extraction, or bias assessment rather than on the inferential properties of alternative synthesis protocols. Early reviews of systematic-review automation emphasized study identification, citation screening, and extraction support (Tsafnat et al., 2014; Marshall & Wallace, 2019). Widely used tools such as Rayyan and ASReview were designed primarily to improve efficiency and transparency in screening and prioritization, not to test whether alternative synthesis procedures lead to different scientific conclusions (Ouzzani et al., 2016; van de Schoot et al., 2021).

The present question therefore sits at the intersection of two literatures. The first concerns research-synthesis automation. That literature establishes the feasibility and value of machine assistance, but it does not address whether a model that updates a cumulative synthesis while reading a corpus behaves differently from one that reviews the same corpus and synthesizes only at the end (Tsafnat et al., 2014; Marshall & Wallace, 2019). The second concerns evaluation of model-generated text. That literature shows that fluency is a poor proxy for evidential adequacy and that long-form outputs require structured human evaluation, preferably at the claim level rather than by holistic impression alone (van der Lee et al., 2019; Krishna et al., 2023).

There is also a direct methodological reason to expect protocol effects. Work on chain-of-thought and zero-shot reasoning has shown that procedural structure can materially alter model behavior even when model weights are held constant (Wei et al., 2022; Kojima et al., 2022). If prompting structure affects reasoning on benchmark tasks, then synthesis structure may likewise affect the kinds of gaps a model can infer from a literature corpus. That possibility is scientifically consequential because literature synthesis is not merely a summarization task. It is a constrained form of comparative reasoning in which absent evidence, contradictory claims, and cross-paper tensions may matter as much as individual-paper content.

The LITRIS environment is suitable for this study because it supports canonical full-text storage, profile-defined semantic dimensions, and deterministic retrieval of document context for downstream analysis (`README.md:3-27`; `docs/portable_dimensions_release_notes_2026-04.md:7-24`; `docs/litris_redesign_summary_2026-03.md:16-39`). Those capabilities make it possible to standardize input packets, preserve provenance, and audit whether a reported gap is genuinely supported by the underlying corpus.

## 2. Pilot Summary and Rationale

The motivating pilot compared batch and accumulative synthesis across five domains using a total of 46 papers. Domain corpora ranged from 8 to 10 papers, and the analysis relied primarily on abstracts retrieved during the same broader analytical effort. The pilot reported 31 gap statements under the batch protocol and 46 under the accumulative protocol, a uniform difference of three additional gaps per domain and an aggregate ratio of 1.48. More importantly, the accumulative protocol appeared to identify categories of problems that were less visible in the batch condition, including contradictions across papers, unexamined governance dependencies, benchmark fragility, and neglected implementation constraints.

These pilot results justify a more rigorous study, but they do not determine its outcome. The pilot did not use blinded evaluation, full-text evidence packets, session isolation, or a preregistered rubric. The present protocol therefore treats the pilot as design guidance rather than as confirmatory evidence. Specifically, the pilot establishes that the question is nontrivial and likely worth testing, while the confirmatory study adopts more conservative effect assumptions than the pilot count contrast would imply.

## 3. Objectives

The primary objective is to determine whether accumulative AI-assisted synthesis yields research-gap statements with higher evidential support and greater agenda-setting utility than those produced by batch synthesis.

Secondary objectives are:

- to determine whether any observed advantage remains after controlling for document order;
- to characterize the types of gaps preferentially surfaced by each protocol;
- to estimate the cost, time, and reviewer burden associated with each protocol;
- to produce a reusable benchmark and evaluation framework for future studies of AI-assisted literature synthesis.

## 4. Research Questions and Hypotheses

### 4.1 Research Questions

**RQ1.** Does ordered accumulative synthesis outperform batch synthesis on expert-rated gap-set utility and claim-level evidential support?

**RQ2.** Does ordered accumulative synthesis produce a different distribution of gap types than batch synthesis?

**RQ3.** Is any observed advantage attributable to cumulative updating itself, to document order, or to both?

**RQ4.** Are protocol effects stable across disciplinary traditions and replicated runs?

### 4.2 Confirmatory Hypotheses

**H1.** Ordered accumulative synthesis will produce higher mean blinded expert ratings of gap-set utility than batch synthesis.

**H2.** Ordered accumulative synthesis will produce a higher proportion of atomic claims judged fully supported by cited evidence than batch synthesis.

**H3.** Ordered accumulative synthesis will yield more cross-paper and structural gap types than batch synthesis.

**H4a.** Shuffled accumulative synthesis will outperform batch synthesis on gap-set utility and claim-level support.

**H4b.** Ordered accumulative synthesis will outperform shuffled accumulative synthesis, indicating that reading order contributes additional information beyond cumulative updating alone.

## 5. Study Design

The study uses an explanatory sequential mixed-methods design with a confirmatory quantitative core. Quantitative analyses will compare synthesis protocols on preregistered outcome measures. Qualitative coding will then analyze the content and structure of the resulting gap statements in order to explain observed quantitative differences and identify failure modes that are not reducible to scalar scores.

The confirmatory design has three mandatory synthesis conditions:

- **Condition A: Batch synthesis.** The model receives all standardized evidence packets for a domain in one prompt and produces a single final synthesis without intermediate synthesis updates.
- **Condition B: Ordered accumulative synthesis.** The model receives the same evidence packets one at a time in a canonical domain order and updates a running synthesis table after each packet.
- **Condition C: Shuffled accumulative synthesis.** The model uses the same update procedure as Condition B, but packet order is randomized within domain.

The confirmatory analysis will use a single preregistered model family in order to isolate protocol effects and avoid an underpowered model-by-protocol design. The present recommendation is OpenAI GPT-5 with `temperature=0` and `top_p=1`, with the exact dated model identifier locked in the OSF registration before confirmatory data collection begins. Cross-family replication is desirable, but it should be treated as a separately registered extension rather than pooled into the primary analysis. The canonical document order for Conditions A and B will be ascending publication year, with ties broken by first-author surname and then title.

## 6. Corpus Construction

### 6.1 Domain Sampling and Domain Questions

The main study will sample eight domains spanning distinct epistemic traditions. The purpose of this stratification is not disciplinary completeness. It is to test whether protocol effects are robust across corpora that differ in evidentiary density, argument structure, and interpretive style.

| Domain | Fixed review question | Core databases | Year window | Search fields |
| ----- | ----- | ----- | ----- | ----- |
| Biomedical or clinical AI | What external-validation and implementation gaps remain in LLM-assisted clinical decision support research? | PubMed, Web of Science | 2019-2026 | title, abstract, keywords |
| Chemistry or materials AI | What reproducibility and validation gaps constrain AI-assisted reaction or materials discovery workflows? | Web of Science, Scopus | 2019-2026 | title, abstract, keywords |
| Energy systems or climate modeling | What benchmark, deployment, and generalization gaps limit machine-learning-based renewable energy or climate prediction studies? | Scopus, Web of Science | 2019-2026 | title, abstract, keywords |
| AI evaluation and reliability | What methodological gaps limit trustworthy evaluation of LLM performance, robustness, or safety? | arXiv, Scopus, Web of Science | 2020-2026 | title, abstract, keywords |
| Policy or public administration | What governance and institutional gaps constrain adoption of AI decision systems in public agencies? | Scopus, Web of Science, JSTOR | 2018-2026 | title, abstract, keywords |
| Education or communication research | What evidence gaps limit claims about the efficacy of AI-supported teaching, feedback, or communication interventions? | ERIC, Scopus, Web of Science | 2018-2026 | title, abstract, keywords |
| Computational social science | What gaps limit causal inference, representational validity, and reproducibility in large-scale digital-trace research? | Scopus, Web of Science | 2018-2026 | title, abstract, keywords |
| Humanities or digital humanities | What methodological, interpretive, and preservation gaps constrain AI-assisted analysis of cultural or historical corpora? | JSTOR, MLA International Bibliography, Web of Science | 2018-2026 | title, abstract, keywords |

### 6.2 Corpus Size and Power Analysis

The target corpus is 30 papers per domain, for a minimum confirmatory corpus of 240 papers. Each domain-condition cell will be replicated in four independent runs, yielding 96 synthesis outputs across the three protocol conditions. Each final synthesis will receive two blinded expert ratings on the primary utility endpoint, for a planned total of 192 utility ratings.

Sample-size planning is driven by the utility outcome rather than by claim-level support, because claim-level scoring will generate substantially more analytical units. A Monte Carlo power analysis will be archived with the preregistration for a linear mixed-effects model with eight domains, three conditions, four runs per condition, and two raters per output. That simulation will be evaluated against the actual confirmatory correction scheme, including the strictest Holm-adjusted threshold of 0.0167 for the first ordered hypothesis test and the 0.025 second-step threshold. If estimated power is inadequate under that correction scheme, the study will be relabeled exploratory or the design will be revised before registration is frozen. The effect assumptions used in that simulation will remain more conservative than the pilot gap-count contrast of 46 versus 31 gaps.

No mid-study sample-size changes will be permitted after the first confirmatory synthesis run. If the design is revised after preregistration but before execution, the revised design will replace the present one in a new dated registration rather than being introduced as an undocumented amendment.

### 6.3 Search and Screening Protocol

Each domain corpus will be assembled using explicit, reproducible search strategies documented at the database level. Search documentation will follow PRISMA-S conventions, and corpus assembly and exclusion reporting will follow PRISMA 2020 conventions as applicable to a methodological comparison rather than to an intervention-effect review. A protocol record will be finalized before the confirmatory corpus is locked.

Searches will be run only in the preregistered databases listed for each domain in Section 6.1. Search dates, query strings, controlled vocabulary, language restrictions, citation-chaining procedures, and de-duplication rules will be archived in the preregistration package.

Title-abstract screening and full-text screening will each be performed independently by two reviewers using a fixed eligibility form. Disagreements will first be resolved by discussion and then, if needed, by a third reviewer. If a title-abstract record is ambiguous but plausibly relevant, it will be retained for full-text screening rather than excluded at the first stage.

### 6.4 Eligibility Criteria

Inclusion criteria:

- peer-reviewed articles or discipline-appropriate archival papers;
- English-language full text available for controlled analysis;
- publication dates within a preregistered window, with limited exceptions for seminal works;
- direct substantive relevance to the domain question.

Exclusion criteria:

- editorials, news items, and opinion pieces without substantive analysis;
- papers lacking accessible full text;
- documents whose substantive content duplicates another included paper;
- papers for which the relevant evidentiary content is too sparse to support gap analysis.

### 6.5 Full-Text Management and Evidence Packets

All included documents will be preserved as canonical full-text snapshots within LITRIS before synthesis begins. Confirmatory synthesis will not use unconstrained full text directly. Instead, each paper will be converted into a standardized evidence packet capped at 2,500 tokens and assembled by scripted deterministic rules from the archived full text. Each packet will include bibliographic metadata, the abstract, author-stated objective or thesis, a methods segment, a findings segment, a limitations segment if present, and up to three verbatim excerpts selected by fixed section-based rules.

Packet assembly rules are fixed in advance. The objective or thesis segment will be drawn from the abstract if an explicit objective, aim, or thesis statement is present; otherwise it will be taken from the first qualifying paragraph of the introduction. The methods segment will be drawn from the first headed methods-like section, with abstract-method sentences used only if no such section exists. The findings segment will be drawn from the first headed results-like or findings-like section, with abstract-results sentences used only as a fallback. The limitations segment will be drawn from a headed limitations section where available, then from the closing discussion if explicit limitations are stated, and otherwise recorded as absent. Verbatim excerpts will be chosen by the scripted packetizer from the retained methods, findings, and limitations segments in that order of priority. No manual editing or generative summarization will be permitted during confirmatory packet construction.

This packetization step serves two purposes. First, it keeps the batch condition within a manageable context window: a 30-paper domain corpus remains below approximately 75,000 content tokens before instructions and output budget are added. Second, it ensures that all conditions receive the same informational substrate. Full text remains available for audit, excerpt recovery, and human scoring.

## 7. Experimental Procedures

### 7.1 Session Isolation and Replication

Each protocol condition will be run in a fresh session. No condition may inherit intermediate hypotheses, hidden state, or summary artifacts from another condition. For each domain-condition pair, the synthesis will be replicated in four independent runs. Separate session identifiers, timestamps, and run metadata will be archived.

### 7.2 Prompt Standardization

All conditions will share a common prompt backbone specifying:

- the domain question under study;
- the definition of a research gap;
- the required output structure;
- the requirement to cite supporting paper identifiers and verbatim excerpts;
- the instruction that unsupported claims are prohibited.

The manipulated feature is synthesis protocol only. The model, packet content, domain question, output limit, and generation settings will otherwise be held constant. Confirmatory runs will use `temperature=0`; any provider that does not expose the required settings will not be used in the confirmatory analysis.

### 7.3 Condition-Specific Procedure

The batch condition will receive all evidence packets in one dossier presented in the same canonical order used by the ordered accumulative condition and then generate a final synthesis.

The ordered accumulative condition will receive one evidence packet at a time. After each packet, the model will update a running synthesis table capped at 12 candidate gaps, each with a short rationale and cited support. After the final packet, the model will compress the table into the required final output.

The shuffled accumulative condition will use the same update procedure, but packet order will be randomized independently for each run.

### 7.4 Output Template

Each final synthesis must contain up to 10 nonredundant gap statements. Each gap entry must contain:

- a short gap title;
- a one- to two-sentence gap description;
- one gap-type label from the preregistered taxonomy;
- one significance statement;
- at least one supporting paper identifier;
- at least one verbatim supporting excerpt.

This template is intended to reduce the risk that apparent protocol effects are driven by verbosity, rhetorical polish, or list inflation.

## 8. Outcome Measures

### 8.1 Co-Primary Outcomes

**Gap-set utility.** Blinded expert raters will score each final synthesis on a seven-point scale in response to the question: *How useful is this gap set for setting a research agenda in this domain?* The confirmatory estimand is the mean difference in utility rating between protocol conditions across synthesized outputs. The unit of analysis is the rater-by-output observation, where one output is a single final synthesis produced for one domain, condition, and run. Scale anchors are fixed in advance:

| Score | Interpretation |
| ----- | -------------- |
| 1 | unusable; materially misleading or vacuous |
| 2 | poor; little agenda-setting value |
| 3 | limited; some relevant observations but weak practical value |
| 4 | mixed; partially useful but uneven or under-supported |
| 5 | useful; clearly serviceable for agenda setting |
| 6 | very useful; strong, well-supported, and actionable |
| 7 | exceptional; publication-grade agenda utility |

**Atomic claim support rate.** Each gap statement will be segmented into atomic claims, defined as the smallest propositions that can be independently supported or contradicted by the cited evidence. For example, a sentence asserting both that a benchmark is overused and that its overuse masks domain shift contains two atomic claims. Two blinded coders will independently segment each gap set; disagreements will be adjudicated before support scoring. Raters will then judge each atomic claim as fully supported, partially supported, unsupported, or contradicted by the cited evidence. The confirmatory estimand is the mean difference in output-level support probability between protocol conditions, where each rated output contributes a binomial summary defined by the number of fully supported atomic claims and the total number of atomic claims in that output. Claim-level annotations will be retained for audit and exploratory analyses, but confirmatory inference will be made on output-level summaries rather than on unaggregated claims.

### 8.2 Secondary Outcomes

- novelty of the gap set relative to routine domain summaries, scored on a five-point anchored scale by blinded raters;
- actionability for agenda setting, funding, or review design, scored on a five-point anchored scale by blinded raters;
- degree of cross-paper integration, scored on a five-point anchored scale indicating whether gaps require synthesis across multiple papers rather than restating isolated findings;
- redundancy, operationalized as the proportion of gap entries flagged by raters as near-duplicates of another gap in the same output;
- evidence traceability, defined as the proportion of gaps for which the cited excerpt and archived source permit direct recovery of evidential support without additional search;
- nonredundant gap count;
- time, token cost, and reviewer effort per condition.

For the four five-point secondary rating scales, score anchors will be uniform across constructs: 1 indicates minimal presence of the named property, 3 indicates mixed or moderate presence, and 5 indicates strong presence. Full wording for each scale will be fixed in the rating codebook deposited with the preregistration.

### 8.3 Exploratory Outcomes

- domain-specific protocol effects;
- lexical or structural markers associated with unsupported inference;
- relationship between packet evidence density and synthesis quality.

## 9. Human Evaluation Procedure

### 9.1 Rater Recruitment

Each domain will be evaluated by two independent domain-competent raters who are blind to synthesis condition and run provenance. A third adjudicating reviewer will resolve substantial disagreements and oversee calibration. At least 25% of all outputs will receive triple independent ratings for reliability estimation. Domain raters must hold either a completed doctorate or advanced doctoral candidacy in the relevant field, plus at least one peer-reviewed publication or equivalent archival research output in that area.

### 9.2 Blinding

Outputs will be anonymized and normalized in presentation. Condition labels, run identifiers, and model provenance will be removed before scoring. Gap order will be randomized during human evaluation to reduce position bias.

### 9.3 Rater Training and Calibration

Raters will receive:

- a codebook defining research gaps, evidential support, novelty, actionability, and redundancy;
- scored exemplar outputs from a separate calibration corpus;
- explicit guidance that rhetorical polish must not substitute for evidential adequacy.

Calibration will occur on a six-output pilot scoring set before confirmatory scoring begins. The confirmatory study will not proceed until the scoring codebook is locked.

### 9.4 Reliability Thresholds

Inter-rater reliability will be quantified using Krippendorff's alpha where scale type permits. The minimum acceptable threshold for the seven-point utility scale is alpha >= 0.67, with alpha >= 0.80 treated as the target for strong reliability. For atomic-claim support coding, the minimum acceptable threshold is alpha >= 0.80 because that coding underwrites a co-primary endpoint. If these thresholds are not met during calibration, the codebook will be revised on the calibration set only. During confirmatory scoring, adjudication will be triggered if utility ratings differ by more than two scale points or if support classifications differ on more than 20% of atomic claims within an output. Outputs lacking two completed primary ratings after replacement-rater procedures will be excluded from confirmatory analyses and listed in a missing-data table. If reliability falls below threshold during confirmatory scoring, the deviation will be reported and the affected analyses will be downgraded to exploratory rather than silently discarded.

## 10. Qualitative Analysis

The qualitative component will examine the kinds of gaps produced by each protocol. The coding taxonomy is fixed in advance:

- empirical gap;
- methodological gap;
- benchmarking or evaluation gap;
- governance or institutional gap;
- infrastructure or deployment gap;
- conceptual or theoretical gap;
- contradiction- or tension-based gap;
- meta-research gap;
- other.

The `other` category is reserved for genuinely residual cases and may not be subdivided inferentially unless an explicit post hoc analysis is labeled as such. Two analysts will double-code at least 25% of gap statements, reconcile differences, and then code the remainder using a shared memo structure. Integration of quantitative and qualitative results will use joint displays and a documented "following a thread" procedure so that distinctive qualitative findings can be traced back to specific quantitative contrasts (O'Cathain et al., 2010).

## 11. Statistical Analysis Plan

The main analysis will use mixed-effects models to account for nesting by domain, synthesis output, and rater.

For the utility outcome, the confirmatory model will be a linear mixed-effects model with protocol condition as the principal fixed effect. The observation entering the model will be one rater-by-output score. Random intercepts will be included for domain, rater, and synthesis output nested within domain. The confirmatory contrasts are:

- ordered accumulative versus batch;
- shuffled accumulative versus batch;
- ordered accumulative versus shuffled accumulative.

For atomic-claim support, the confirmatory model will be a generalized linear mixed-effects model with a binomial link, where the observation entering the model is the output-level count of fully supported atomic claims out of the total number of atomic claims scored for that output. Random intercepts will be included for domain, synthesis output, and rater.

Gap-type distributions will be analyzed descriptively and, where cell counts permit, with multinomial or log-linear models as secondary analyses. Nonredundant gap counts will be modeled with Poisson or negative binomial links depending on dispersion diagnostics.

Multiplicity across the three confirmatory contrasts will be controlled with a preregistered Holm correction. Effect sizes and confidence intervals will be reported for all inferential analyses. Sensitivity analyses will include:

- treating partial support as support for a relaxed evidential endpoint;
- excluding residual `other` taxonomy cases;
- re-estimating models after removal of any synthesis outputs with incomplete provenance.

## 12. Open Science and Preregistration Plan

The protocol will be preregistered on the Open Science Framework before confirmatory corpus assembly is finalized and before any confirmatory synthesis runs are executed. The registration package will include:

- research questions and hypotheses;
- domain questions, eligibility criteria, and search strategies;
- packet-construction rules and packet-length caps;
- prompts, system instructions, and model settings;
- randomization procedures and session-isolation rules;
- outcome definitions, rating codebooks, and reliability thresholds;
- the statistical analysis plan;
- deviation logging procedures.

All nonrestricted study materials should be made public at manuscript submission or earlier. If licensing, reviewer blinding, or legal access constraints require temporary restriction, a private OSF registration with time-stamped materials should be created first and then released publicly at the earliest permissible point.

The protocol is suitable for submission as a Stage 1 Registered Report. If that route is pursued, no confirmatory corpus lock, synthesis execution, or confirmatory scoring may begin before in-principle acceptance unless the study is explicitly reframed as a standard preregistered article.

## 13. Data Management and Reproducibility

The study will maintain separate archives for:

- raw search results and de-duplication logs;
- screening decisions and exclusion reasons;
- canonical full-text snapshots and packet manifests;
- prompts, model outputs, and run metadata;
- rater assignments, score sheets, claim-segmentation records, and adjudication notes;
- analysis scripts and rendered result tables.

All manuscript figures and tables must be reproducible from archived scripts. Derived public materials should be deposited in a stable repository with versioned checksums.

## 14. Ethical and Practical Considerations

This study poses minimal human-subject risk, but expert raters are still human participants for the purposes of workload, consent, and confidentiality. Institutional review requirements should therefore be checked before recruitment begins.

The primary non-human ethics issues concern intellectual property and quotation. Full texts should be used under lawful access conditions, and any quoted excerpts in the manuscript should remain limited, necessary, and traceable.

Potential conflicts of interest include prior commitments to one synthesis workflow, financial or institutional ties to model providers, and domain-expert allegiance to particular theoretical schools. These should be disclosed and, where possible, balanced across raters.

## 15. Feasibility and Timeline

A realistic first-cycle timeline is 14 to 18 weeks.

- Weeks 1-3: finalize protocol, register the study, complete calibration scoring, and validate search strategies.
- Weeks 4-7: assemble and screen corpora, archive full texts, construct packets, and lock the confirmatory dataset.
- Weeks 8-10: run synthesis conditions and normalize outputs for blinded review.
- Weeks 11-14: complete blinded human evaluation, claim segmentation, adjudication, and reliability checks.
- Weeks 15-18: execute quantitative and qualitative analyses, write the manuscript, and prepare the public archive.

## 16. Expected Contributions

If successful, the study will make four contributions.

First, it will provide a controlled estimate of whether synthesis protocol materially affects AI-assisted gap identification. Second, it will separate verbosity effects from evidential and integrative gains. Third, it will contribute a reusable evaluation framework for long-form AI synthesis that combines claim-level support assessment with expert utility judgments. Fourth, it will demonstrate how a literature-indexing environment such as LITRIS can support transparent, auditable, and domain-spanning studies of AI-assisted scholarship.

## 17. Anticipated Limitations

Even a strengthened design will not eliminate all uncertainty. Expert judgment remains partly interpretive, domain coverage will still be selective, and model updates may complicate strict replication over time. The packetization step, while necessary for protocol parity, is itself a design choice that may suppress some forms of contextual nuance. In addition, models may have encountered some included papers during pretraining, which introduces a potential leakage confound that cannot be fully eliminated even when outputs are grounded in provided packets. These limitations should be reported as boundary conditions rather than minimized in discussion prose.

## References

Chambers, C. D. (2013). Registered reports: A new publishing initiative at Cortex. *Cortex, 49*(3), 609-610. https://doi.org/10.1016/j.cortex.2012.12.016

Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., & Iwasawa, Y. (2022). Large language models are zero-shot reasoners. *arXiv*. https://doi.org/10.48550/arXiv.2205.11916

Krippendorff, K. (2011). *Computing Krippendorff's alpha-reliability* (Working Paper No. 43). Annenberg School for Communication, University of Pennsylvania. https://repository.upenn.edu/handle/20.500.14332/2089

Krishna, K., Bransom, E., Kuehl, B., Iyyer, M., Dasigi, P., Cohan, A., & Lo, K. (2023). LongEval: Guidelines for human evaluation of faithfulness in long-form summarization. In *Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics* (pp. 1650-1669). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.eacl-main.121

Marshall, I. J., & Wallace, B. C. (2019). Toward systematic review automation: A practical guide to using machine learning tools in research synthesis. *Systematic Reviews, 8*(1), 163. https://doi.org/10.1186/s13643-019-1074-9

Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration revolution. *Proceedings of the National Academy of Sciences of the United States of America, 115*(11), 2600-2606. https://doi.org/10.1073/pnas.1708274114

O'Cathain, A., Murphy, E., & Nicholl, J. (2008). The quality of mixed methods studies in health services research. *Journal of Health Services Research & Policy, 13*(2), 92-98. https://doi.org/10.1258/jhsrp.2007.007074

O'Cathain, A., Murphy, E., & Nicholl, J. (2010). Three techniques for integrating data in mixed methods studies. *BMJ, 341*, c4587. https://doi.org/10.1136/bmj.c4587

Ouzzani, M., Hammady, H., Fedorowicz, Z., & Elmagarmid, A. (2016). Rayyan-a web and mobile app for systematic reviews. *Systematic Reviews, 5*(1), 210. https://doi.org/10.1186/s13643-016-0384-4

Page, M. J., McKenzie, J. E., Bossuyt, P. M., Boutron, I., Hoffmann, T. C., Mulrow, C. D., Shamseer, L., Tetzlaff, J. M., Akl, E. A., Brennan, S. E., Chou, R., Glanville, J., Grimshaw, J. M., Hróbjartsson, A., Lalu, M. M., Li, T., Loder, E. W., Mayo-Wilson, E., McDonald, S., ... Moher, D. (2021). The PRISMA 2020 statement: An updated guideline for reporting systematic reviews. *BMJ, 372*, n71. https://doi.org/10.1136/bmj.n71

Rethlefsen, M. L., Kirtley, S., Waffenschmidt, S., Ayala, A. P., Moher, D., Page, M. J., Koffel, J. B., & PRISMA-S Group. (2021). PRISMA-S: An extension to the PRISMA statement for reporting literature searches in systematic reviews. *Journal of the Medical Library Association, 109*(2), 174-200. https://doi.org/10.5195/jmla.2021.962

Shamseer, L., Moher, D., Clarke, M., Ghersi, D., Liberati, A., Petticrew, M., Shekelle, P., Stewart, L. A., & PRISMA-P Group. (2015). Preferred reporting items for systematic review and meta-analysis protocols (PRISMA-P) 2015: Elaboration and explanation. *BMJ, 349*, g7647. https://doi.org/10.1136/bmj.g7647

Tsafnat, G., Glasziou, P., Choong, M. K., Dunn, A., Galgani, F., & Coiera, E. (2014). Systematic review automation technologies. *Systematic Reviews, 3*(1), 74. https://doi.org/10.1186/2046-4053-3-74

van de Schoot, R., de Bruin, J., Schram, R., Zahedi, P., de Boer, J., Weijdema, F., Kramer, B., Huijts, M., Hoogerwerf, M., Ferdinands, G., Harkema, A., Willemsen, J., Ma, Y., Fang, Q., Hindriks, S., Tummers, L., & Oberski, D. L. (2021). An open source machine learning framework for efficient and transparent systematic reviews. *Nature Machine Intelligence, 3*(2), 125-133. https://doi.org/10.1038/s42256-020-00287-7

van der Lee, C., Gatt, A., van Miltenburg, E., Wubben, S., & Krahmer, E. (2019). Best practices for the human evaluation of automatically generated text. In *Proceedings of the 12th International Conference on Natural Language Generation* (pp. 355-368). Association for Computational Linguistics. https://doi.org/10.18653/v1/W19-8643

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. *arXiv*. https://doi.org/10.48550/arXiv.2201.11903
