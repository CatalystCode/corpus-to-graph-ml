#!/usr/bin/python
# -*- coding: utf-8 -*-

from __init__ import validate

total = 0
errors = 0

def testValidate(value, expected, index):
    global total
    global errors
    total+=1
    try:
        result = validate(value)
        result = [value['mirna'] for value in result['detectedMirnas']]
        assert result == expected
    except AssertionError as e:
        errors+=1
        print index, ': >>> ', result, 'not equals to', expected


testValidate("""Validation assays confirmed the dysregulation of miR-223, 
    miR-146a and miR-155 previously associated with human rheumatoid 
    arthritis (RA) pathology, as well as that of miR-221/222 and 
    miR-323-3p.""",
    ['hsa-mir-223', 'hsa-mir-146a', 'hsa-mir-155', 'hsa-mir-221', 'hsa-mir-222', 'hsa-mir-323-3p'],
    0)

testValidate("""We found that both miR-17 and miR-20a (miR-17/20a) target 
    the UBE2C gene in gastric cancer cells.""",
    ['hsa-mir-17', 'hsa-mir-20a', 'hsa-mir-17', 'hsa-mir-20a'],
    1)

testValidate("""Collectively, these data identify the E2F1/miR-421/Pink 
    axis as a regulator of mitochondrial fragmentation and cardiomyocyte 
    apoptosis, and suggest potential therapeutic targets in treatment of 
    cardiac diseases.""",
    ['hsa-mir-421'],
    2)

testValidate("""Based on extant data linking MIR 137 gene with structural 
    brain anomalies and functional brain activations in schizophrenia, 
    we hypothesized that MIR137 risk variants rs1625579 and rs1198588 
    would be associated with reduced fractional anisotropy in 
    frontostriatal brain regions, impaired neurocognitive functioning 
    and worse psychotic symptoms in schizophrenia patients compared with 
    healthy controls.""",
    ['hsa-mir-137', 'hsa-mir-137'],3)

testValidate("""Betulinic acid-dependent repression of Sp1, Sp3, Sp4, 
    and Sp-regulated genes was due, in part, to induction of the Sp 
    repressor ZBTB10 and downregulation of microRNA-27a (miR-27a), which 
    constitutively inhibits ZBTB10 expression, and we show for the first 
    time that the effects of betulinic acid on the miR-27a:ZBTB10-Sp 
    transcription factor axis were cannabinoid 1 (CB1) and CB2 
    receptor-dependent, thus identifying a new cellular target for this 
    anticancer agent.""",
    ['hsa-mir-27a', 'hsa-mir-27a', 'hsa-mir-27a'],4)


testValidate("""The reintroduction of miR-148a and miR-34b/c in cancer 
    cells with epigenetic inactivation inhibited their motility, reduced 
    tumor growth, and inhibited metastasis formation in xenograft models, 
    with an associated down-regulation of the miRNA oncogenic target 
    genes, such as C-MYC, E2F3, CDK6, and TGIF2.""",
    ['hsa-mir-148a', 'hsa-mir-34b', 'hsa-mir-34c'],5)


testValidate("""TGF), known to inhibit mir-200c expression in tumour 
    cells , along with feedforward and negative feedback loops between the 
    miR-200/ZEB1/JAG1 (; ; this study) would establish a vicious circle' in 
    the metastatic bone microenvironment (model ), as may also apply to the 
    organ tropism of other cancers.""",
    ['hsa-mir-200c', 'hsa-mir-200'],6)


testValidate("""The roles of miR-17-5p and p21 were evaluated with 
    specific antisense oligonucleotides (ODN) that were designed and 
    used to inhibit their expression.""",
    ['hsa-mir-17-5p'],7)


testValidate("""Consistently, depletion of miR-26a/b by miR-26 sponge 
    could increase the activity of luciferase reporter genes fused to 
    the 3 UTR of the same cohort of nine genes (AGPAT5, CHD1, ERLIN1, 
    GREB1, HSPA8, KPNA2, MREG, NARG1 and PLOD2) by more than 30% 
    (Figured).""",
    ['hsa-mir-26a', 'hsa-mir-26b', 'hsa-mir-26'],8)


testValidate("""In our present study, we found that the expression of 
    miR-361-5p in CRPC was lower than in androgen-dependent prostate 
    cancer (ADPC), indicating that miR-361-5p may play an important role 
    in the progression of ADPC to CRPC.""",
    ['hsa-mir-361-5p', 'hsa-mir-361-5p'],9)


testValidate("""The expression of miR-146a, CD40, CD80 and CD86 on AchR 
    specific B cells were analyzed by qRT-PCR and flow cytometry.""",
    ['hsa-mir-146a'],10)


testValidate("""METHODS: We examined the association between the 
    expression of miR-16, miR-21, miR-93, miR-135b, miR-146a, and miR-182 
    in total RNA from the placentas of 86 term infants as measured by 
    quantitative real-time PCR and newborn neurobehavioral outcomes as 
    assessed using the NICU Network Neurobehavioral Scales (NNNS).""",
    ['hsa-mir-16', 'hsa-mir-21', 'hsa-mir-93', 'hsa-mir-135b', 'hsa-mir-146a', 'hsa-mir-182'],
    11)


testValidate("""METHODS: The current study validates nine miRNAs 
    (miR-18a/b miR-25, miR-29c, miR-106b, miR375, miR-424, miR-505 and 
    let-7b) significantly correlated with established prognostic 
    breast cancer biomarkers.""",
    ['hsa-mir-18a', 'hsa-mir-18b', 'hsa-mir-25', 'hsa-mir-29c', 'hsa-mir-106b', 'hsa-mir-375', 
    'hsa-mir-424', 'hsa-mir-505', 'hsa-let-7b'],12)


testValidate("""No significant relationships were observed between these 
    two single nucleotide polymorphisms (SNPs) and onset risk of HCC 
    after adjusting the factors as age, gender, smoking and drinking 
    status in comparison with HBsAg positive controls: hsa-mir-146a 
    rs2910164 (CG + GG vs CC): adjusting OR = 1.10, 95%CI: 0.90 - 1.36; 
    hsa-mir-196-a2 rs11614913 (CC + CT vs TT): adjusting OR = 1.01, 
    95%CI: 0.81 - 1.25; as well as in comparison with HBsAg negative 
    controls: hsa-mir-146a rs2910164 (CG + GG vs CC): adjusting OR = 
    1.06, 95%CI: 0.87 - 1.29; hsa-mir-196-a2 rs11614913 (CC + CT vs TT): 
    adjusting OR = 0.94, 95%CI: 0.76 - 1.16.""",
    ['hsa-mir-146a', 'hsa-mir-196-a2', 'hsa-mir-146a', 'hsa-mir-196-a2'],
    13)


testValidate("""Hsa-miR-96 caused a decrease in SOX5 3-UTR luciferase 
    activity by 60.34%4.79%, and both hsa-miR-7 and hsa-miR-17 caused 
    a decrease in NR4A3 3-UTR luciferase activity by 65.01%4.07% and 
    45.11%6.76, respectively, compared with controls (FigureC).""",
    ['hsa-mir-96', 'hsa-mir-7', 'hsa-mir-17'],14)


testValidate("""Furthermore, miR-20a and miR-17-5p were increased in 
    the metastatic carcinoma and six atypical pituitary adenomas as 
    compared to eight typical pituitary adenomas as measured by 
    quantitative real-time PCR.""",
    ['hsa-mir-20a', 'hsa-mir-17-5p'],15)


testValidate("""These results suggested that anti-miR-33a inhibit 
    activation and extracellular matrix production, at least in part, 
    via the activation of PI3K/Akt pathway and PPAR-a and anti sense 
    of miR-33a may be a novel potential therapeutic approach for 
    treating hepatic fibrosis in the future.""",
    ['hsa-mir-33a'],16)


testValidate("""The results suggested that the miR-181b, miR-219-2-3p, 
    miR-346, miR-195, miR-1308, miR-92a, miR-17, miR-103 and let-7g are 
    the key players to reflect the schizophrenia illnesses status and 
    may serve as candidate biomarkers for diagnosis of schizophrenia.""",
    ['hsa-mir-181b', 'hsa-mir-219-2-3p', 'hsa-mir-346', 'hsa-mir-195', 'hsa-mir-1308',
    'hsa-mir-92a', 'hsa-mir-17', 'hsa-mir-103', 'hsa-let-7g'],17)


testValidate("""Specifically discussed miRs include miR-7, miR-9/miR-9*, 
    miR-10a/miR-10a*/miR-10b, miR-15b, miR-17-92, miR-21, miR-26a, 
    miR-34a, miR-93, miR-101, miR-124, miR-125a, miR-125b, miR-128, 
    miR-137, miR-146b-5p, miR-153, miR-181a/miR-181b, miR-196a/miR-196b, 
    miR-218, miR-221/miR-222, miR-296, miR-302-367, miR-326, miR-381, 
    miR-451, and let-7a.""",
    ['hsa-mir-7', 'hsa-mir-9', 'hsa-mir-9*', 'hsa-mir-10a', 'hsa-mir-10a*', 'hsa-mir-10b', 
    'hsa-mir-15b', 'hsa-mir-17-92', 'hsa-mir-21', 'hsa-mir-26a', 'hsa-mir-34a', 'hsa-mir-93', 
    'hsa-mir-101', 'hsa-mir-124', 'hsa-mir-125a', 'hsa-mir-125b', 'hsa-mir-128', 'hsa-mir-137', 
    'hsa-mir-146b-5p', 'hsa-mir-153', 'hsa-mir-181a', 'hsa-mir-181b', 'hsa-mir-196a', 
    'hsa-mir-196b', 'hsa-mir-218', 'hsa-mir-221', 'hsa-mir-222', 'hsa-mir-296', 'hsa-mir-302-367', 
    'hsa-mir-326', 'hsa-mir-381', 'hsa-mir-451', 'hsa-let-7a'],18)


testValidate("""TT genotype for miR-196a2 gene also showed 3.2-fold 
    risk toward LC and the risk was fivefold higher for squamous cell 
    carcinoma.""",
    ['hsa-mir-196a2'],19)


testValidate("""Thus, loss of miR-125b-1 may have a key role in the 
    pathogenesis and progression of squamous cell carcinomas of head 
    and neck and possibly of other tumors.""",
    ['hsa-mir-125b-1'],20)


testValidate("""The present prospective case-control study investigated 
    the involvement of microRNA (miR)-10b in the development of bone 
    metastasis arising from primary breast carcinoma.""",
    ['hsa-mir-10b'],21)


testValidate("""Four other miRNAs (miR-146b, -181b, let-7a and let-7c) 
    are known oncogenic or tumor suppressor miRNAs.""",
    ['hsa-mir-146b', 'hsa-mir-181b', 'hsa-let-7a', 'hsa-let-7c'],22)


testValidate("""BACKGROUND: The purpose of this study was to identify 
    new tumour suppressor microRNAs (miRs) in clear cell renal cell 
    carcinoma (ccRCC), carry out functional analysis of their suppressive 
    role and identify their specific target genes.""",
    [],23)


testValidate("""Subsequent quantitative PCR analyses of these splenic 
    B cells revealed that C/EBPb, a transcriptional regulator of 
    interleukin-6 that is linked to B-cell lymphoproliferative 
    disorders, is downregulated when either miR-K12-11 or miR-155 is 
    ectopically expressed.""",
    ['hsa-mir-K12-11', 'hsa-mir-155'],24)


testValidate("""Thus, there is a possibility that the lack of change 
    in miRs-182 and -96 following acoustic trauma is due to a slower 
    degradation rate or no degradation compared to the targeted degradation 
    of miR-183, which in turn may lead to the inconsistent expression 
    pattern of these miRNAs within the cluster.""",
    ['hsa-mir-182', 'hsa-mir-96', 'hsa-mir-183'],25)


testValidate("""Hsa-miR-92b and hsa-miR-9/9* were reported previously 
    to be expressed in brain tumors and in cell lines derived from 
    brain tumors and were documented to be expressed specifically in 
    the developing nervous system """,
    ['hsa-mir-92b', 'hsa-mir-9', 'hsa-mir-9*'],26)


testValidate("""However, only 3 miRNAs (miR-199a-5p, -27a, and -29a) 
    correlated with hypertrophy; more importantly, only miR-29a 
    correlated also with fibrosis.""",
    ['hsa-mir-199a-5p', 'hsa-mir-27a', 'hsa-mir-29a', 'hsa-mir-29a'],27)


testValidate("""We found that target genes such as CDH1 (miR-1/206), 
    ATM (miR-18a/b), KLF6 (miR-18a/b and miR-181c), Smad2(miR-18a/b, 
    miR-1/206 and miR-149), Dicer were down expressed with the 
    development of NPC, while BCL2L2 (miR-29a/b/c and miR-203), and YY1 
    (miR-29a/b/c) were overexpressed during the development of NPC.""",
    ['hsa-mir-1', 'hsa-mir-206', 'hsa-mir-18a', 'hsa-mir-18b', 'hsa-mir-18a', 'hsa-mir-18b', 
    'hsa-mir-181c', 'hsa-mir-18a', 'hsa-mir-18b', 'hsa-mir-1', 'hsa-mir-206', 'hsa-mir-149',
    'hsa-mir-29a', 'hsa-mir-29b', 'hsa-mir-29c', 'hsa-mir-203', 'hsa-mir-29a', 'hsa-mir-29b',
    'hsa-mir-29c'],28)


testValidate("""The miR-200 family (miR-200a, -200b, -200c, -141 and -429) 
    and miR-205 are frequently silenced in advanced cancer and have been 
    implicated in epithelial to mesenchymal transition (EMT) and tumor 
    invasion by targeting the transcriptional repressors of E-cadherin, 
    ZEB1 and ZEB2.""",
    ['hsa-mir-200', 'hsa-mir-200a', 'hsa-mir-200b', 'hsa-mir-200c', 'hsa-mir-141', 'hsa-mir-429',
    'hsa-mir-205'],29)


testValidate("""Here, the expression of the miRNAs miR-15a/16-1 in 
    PBMC, CD4, and CD8 from RR-MS patients has been investigated.""",
    ['hsa-mir-15a', 'hsa-mir-16-1'],30)


testValidate("""Subsequent quantitative PCR analyses of these splenic 
    B cells revealed that C/EBPb, a transcriptional regulator of 
    interleukin-6 that is linked to B-cell lymphoproliferative 
    disorders, is downregulated when either miR-K12-11 is 
    ectopically expressed.""",
    ['hsa-mir-K12-11'], 31)

print total-errors, '/',total, (total-errors)*100/total, '% test passed'