# Biomedical MCP Benchmark Report

## Overview
- New server: https://tooluniverse-smcp.onrender.com
- Old server: https://tooluniversemcpserver.onrender.com
- Sample size: 50
- Top-k: 5

## Workflow Quality Metrics

### NEW
- selection_hit_at_k: 0.48
- selection_top1_rate: 0.24
- execution_valid_rate: 1.0
- case_chain_all_steps_top1_rate: 0.0
- case_chain_all_steps_hit_rate: 0.125
- chain_similarity_hit_at_k: 0.5
- persona_chain_hit_at_k: 1.0

### OLD
- selection_hit_at_k: 0.48
- selection_top1_rate: 0.24
- execution_valid_rate: 1.0
- case_chain_all_steps_top1_rate: 0.0
- case_chain_all_steps_hit_rate: 0.125
- chain_similarity_hit_at_k: None
- persona_chain_hit_at_k: None

## Case Studies
- app_uniprot_literature: APP protein function and literature
  - context: Validate APP protein annotations and link to literature evidence.
  - query: Swiss-Prot entry for amyloid beta precursor protein (APP) human accession
  - expected_tools: ['UniProt_get_entry_by_accession']
  - query: Functional annotation of APP protein from UniProt (neuronal/synaptogenesis)
  - expected_tools: ['UniProt_get_function_by_accession']
  - query: Literature on amyloid beta precursor protein and Alzheimer disease
  - expected_tools: ['EuropePMC_search_articles']
- egfr_cellular_context: EGFR cellular context
  - context: Combine HPA and Open Targets to validate EGFR localization, ontology, and interactions.
  - query: Human Protein Atlas basic gene info for EGFR by Ensembl ID
  - expected_tools: ['HPA_get_gene_basic_info_by_ensembl_id']
  - query: Subcellular location for EGFR protein
  - expected_tools: ['HPA_get_subcellular_location']
  - query: EGFR protein interaction partners
  - expected_tools: ['HPA_get_protein_interactions_by_gene', 'OpenTargets_get_target_interactions_by_ensemblID']
  - query: Gene ontology terms for EGFR target
  - expected_tools: ['OpenTargets_get_target_gene_ontology_by_ensemblID']
- lung_carcinoma_targets: Lung carcinoma target discovery
  - context: Use Open Targets to prioritize lung carcinoma targets and verify evidence with literature.
  - query: Targets associated with lung carcinoma (EFO_0001071)
  - expected_tools: ['OpenTargets_get_associated_targets_by_disease_efoId']
  - query: Interaction partners for EGFR target
  - expected_tools: ['OpenTargets_get_target_interactions_by_ensemblID']
  - query: Literature evidence for EGFR lung carcinoma
  - expected_tools: ['EuropePMC_search_articles']
- imatinib_moa_analogs: Imatinib mechanism and analogs
  - context: Profile imatinib mechanism of action, linked diseases, and similar molecules.
  - query: Mechanism of action for imatinib by ChEMBL ID
  - expected_tools: ['OpenTargets_get_drug_mechanisms_of_action_by_chemblId']
  - query: Diseases associated with imatinib
  - expected_tools: ['OpenTargets_get_associated_diseases_by_drug_chemblId']
  - query: Find molecules similar to imatinib in ChEMBL
  - expected_tools: ['ChEMBL_search_similar_molecules']
- warfarin_safety_triangulation: Warfarin safety triangulation
  - context: Compare label adverse reactions with FAERS counts and DailyMed SPL metadata.
  - query: FDA label adverse reactions for warfarin
  - expected_tools: ['FDA_get_adverse_reactions_by_drug_name']
  - query: FAERS reaction counts for warfarin
  - expected_tools: ['FAERS_count_reactions_by_drug_event']
  - query: DailyMed SPLs for warfarin
  - expected_tools: ['DailyMed_search_spls']
- breast_cancer_gwas_targets: Breast cancer GWAS evidence and targets
  - context: Connect GWAS associations to gene-level SNPs and Open Targets evidence.
  - query: GWAS associations for breast cancer
  - expected_tools: ['gwas_get_associations_for_trait']
  - query: SNPs mapped to BRCA1
  - expected_tools: ['gwas_get_snps_for_gene']
  - query: Targets linked to breast carcinoma (EFO_0000305)
  - expected_tools: ['OpenTargets_get_associated_targets_by_disease_efoId']
- apoptosis_pathway_literature: Caspase-3 apoptosis pathway
  - context: Validate apoptosis function, pathway reactions, and supporting literature.
  - query: UniProt function for caspase-3 (CASP3) apoptosis
  - expected_tools: ['UniProt_get_function_by_accession']
  - query: Reactome pathway reactions for apoptosis (R-HSA-109581)
  - expected_tools: ['Reactome_get_pathway_reactions']
  - query: Literature for caspase-3 apoptosis
  - expected_tools: ['EuropePMC_search_articles']
- pdb_to_uniprot_mapping: PDB to UniProt mapping (adenylate kinase)
  - context: Validate PDB sequence extraction and map to UniProt entry for E. coli adenylate kinase.
  - query: Retrieve protein sequence from PDB ID 1AKE
  - expected_tools: ['get_sequence_by_pdb_id']
  - query: Retrieve UniProt sequence for adenylate kinase (P69441)
  - expected_tools: ['UniProt_get_sequence_by_accession']
  - query: Retrieve UniProt entry for adenylate kinase (P69441)
  - expected_tools: ['UniProt_get_entry_by_accession']