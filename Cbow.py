"""
Hybrid Lexical Simplification Pipeline

Steps:
1. Generate semantic neighbors using CBOW (Word2Vec)
2. Filter candidates using word frequency (simplicity filter)
3. Re-rank top candidates using DistilBERT MLM probability
4. Replace best candidate in sentence
"""

import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from gensim.models import KeyedVectors
import gensim.downloader as api
from wordfreq import zipf_frequency

# ===============================
# CONFIGURATION
# ===============================

TOP_CANDIDATES_CBow = 20
TOP_CANDIDATES_AFTER_FILTER = 5
ALPHA = 0.7  # weight for contextual score
BETA = 0.3   # weight for simplicity score

# ===============================
# LOAD MODELS
# ===============================

print("\n🔹 Loading Word2Vec (CBOW) model...")
# This will download the model if it's not already downloaded
w2v_model = api.load("word2vec-google-news-300")

print("✔ Word2Vec loaded")

print("\n🔹 Loading DistilBERT MLM...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
mlm_model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlm_model.to(device)
mlm_model.eval()

print("✔ DistilBERT loaded\n")


# ===============================
# STEP 1: CBOW Candidate Generation
# ===============================

# def generate_cbow_candidates(word):
#     print(f"\n🔹 Generating CBOW neighbors for: '{word}'")

#     if word not in w2v_model:
#         print("⚠ Word not in vocabulary")
#         return []

#     neighbors = w2v_model.most_similar(word, topn=TOP_CANDIDATES_CBow)

#     candidates = [w for w, _ in neighbors]

#     print("Top CBOW candidates:")
#     print(candidates)

#     return candidates

def generate_cbow_candidates(word):
    print(f"\n🔹 Generating CBOW neighbors for: '{word}'")

    if word not in w2v_model:
        print("⚠ Word not in vocabulary")
        return []

    neighbors = w2v_model.most_similar(word, topn=TOP_CANDIDATES_CBow)

    print("\nTop CBOW candidates with similarity scores:")
    for w, score in neighbors:
        print(f"{w:<25} | Cosine Similarity = {score:.4f}")

    return neighbors



# ===============================
# STEP 2: Frequency Filtering
# ===============================

def filter_by_frequency(original_word, candidates):
    print("\n🔹 Filtering by simplicity (frequency)...")

    original_freq = zipf_frequency(original_word, "en")
    print(f"Original word frequency: {original_freq:.2f}")

    simpler_candidates = []

    for word in candidates:
        freq = zipf_frequency(word, "en")

        # Keep only words that are more frequent (simpler)
        if freq > original_freq:
            simpler_candidates.append((word, freq))

    # Sort by frequency descending
    simpler_candidates.sort(key=lambda x: x[1], reverse=True)

    # Keep top 5
    simpler_candidates = simpler_candidates[:TOP_CANDIDATES_AFTER_FILTER]

    print("Filtered simpler candidates:")
    for word, freq in simpler_candidates:
        print(f"{word} (freq={freq:.2f})")

    return [word for word, _ in simpler_candidates]


# ===============================
# STEP 3: DistilBERT Re-ranking
# ===============================

def rerank_with_mlm(sentence, target_word, candidates):
    print("\n🔹 Re-ranking with DistilBERT MLM...")

    masked_sentence = sentence.replace(target_word, tokenizer.mask_token, 1)
    inputs = tokenizer(masked_sentence, return_tensors="pt").to(device)

    mask_index = torch.where(
        inputs["input_ids"] == tokenizer.mask_token_id
    )[1]

    with torch.no_grad():
        outputs = mlm_model(**inputs)

    logits = outputs.logits
    mask_logits = logits[0, mask_index, :]
    probs = F.softmax(mask_logits, dim=-1)

    ranked = []

    for word in candidates:
        token_id = tokenizer.convert_tokens_to_ids(word)

        # Skip multi-token words
        if token_id == tokenizer.unk_token_id:
            continue

        lm_score = probs[0, token_id].item()
        simplicity_score = zipf_frequency(word, "en")

        final_score = ALPHA * lm_score + BETA * simplicity_score

        ranked.append((word, lm_score, simplicity_score, final_score))

    ranked.sort(key=lambda x: x[3], reverse=True)

    print("\nRe-ranked candidates:")
    for word, lm, simp, final in ranked:
        print(f"{word} | LM={lm:.4f} | Simplicity={simp:.2f} | Final={final:.4f}")

    return ranked


# ===============================
# STEP 4: Replace Best Candidate
# ===============================

def simplify_sentence(sentence, target_word):
    print("\n====================================")
    print("Original sentence:", sentence)

    cbow_candidates = generate_cbow_candidates(target_word)

    if not cbow_candidates:
        print("⚠ No CBOW candidates found")
        return sentence

    filtered_candidates = filter_by_frequency(target_word, cbow_candidates)

    if not filtered_candidates:
        print("⚠ No simpler candidates found")
        return sentence

    ranked_candidates = rerank_with_mlm(sentence, target_word, filtered_candidates)

    if not ranked_candidates:
        print("⚠ No valid MLM-ranked candidates")
        return sentence

    best_word = ranked_candidates[0][0]

    simplified = sentence.replace(target_word, best_word, 1)

    print("\n✅ Final Simplified Sentence:")
    print(simplified)

    return simplified


# ===============================
# TEST EXAMPLE
# ===============================

# sentence = "The culprit was apprehended."
# target_word = "apprehended"

sentences = [
    ("The culprit was apprehended.", "apprehended"),
    ("The project was abruptly terminated.", "terminated"),
    ("She exhibited remarkable resilience.", "remarkable"),
    ("The corporation initiated a comprehensive restructuring process.","comprehensive"),
    ("His intentions were ambiguous.","His intentions were ambiguous."),
    ("We must evacuate the premises immediately.","evacuate"),
    ("We must evacuate the premises immediately.","premises"),
    ("That explanation is completely absurd.","absurd"),
    ("I cannot tolerate this outrageous behavior.","tolerate"),
    ("The scientist formulated a groundbreaking hypothesis.","groundbreaking"),
    ("The scientist formulated a groundbreaking hypothesis.","hypothesis"),
    ("The project was abruptly terminated due to financial constraints.","abruptly"),
    ("The committee unanimously endorsed the controversial legislation.","endorsed"),
    ("The committee unanimously endorsed the controversial legislation.","legislation")
]

for sentence, target_word in sentences:
    simplify_sentence(sentence,target_word)

# simplify_sentence(sentence, target_word)


### Original sentence: The culprit was apprehended.

🔹 Generating CBOW neighbors for: 'apprehended'
Top CBOW candidates:
['arrested', 'apprehend', 'detained', 'nabbed', 'arrrested', 'rearrested', 'arested', 'apprehending', 'arrest', 'surrendered_peacefully', 'ar_rested', 'allegedly_burglarizing', 'fled', 'police', 'custody', 'accosted', 'allegedly_assaulted', 'arresting', 'absconded', 'suspects']

🔹 Filtering by simplicity (frequency)...
Original word frequency: 3.31
Filtered simpler candidates:
police (freq=5.33)
arrested (freq=4.64)
arrest (freq=4.55)
custody (freq=4.22)
fled (freq=4.05)

🔹 Re-ranking with DistilBERT MLM...

Re-ranked candidates:
police | LM=0.0004 | Simplicity=5.33 | Final=1.5993
arrested | LM=0.0111 | Simplicity=4.64 | Final=1.3998
arrest | LM=0.0004 | Simplicity=4.55 | Final=1.3652
custody | LM=0.0001 | Simplicity=4.22 | Final=1.2661
fled | LM=0.0001 | Simplicity=4.05 | Final=1.2151

✅ Final Simplified Sentence:
The culprit was police.

====================================
Original sentence: The project was abruptly terminated.

🔹 Generating CBOW neighbors for: 'terminated'
Top CBOW candidates:
['terminate', 'terminating', 'termination', 'terminates', 'unilaterally_terminated', 'wrongfully_terminated', 'rehired', 'rescinded', 're_instated', 'revoked', 'constructively_discharged', 'suspended', 'unilaterally_terminate', 'reinstated', 'reassigned', 'Terminated', 'mutually_terminated', 'withdrawn', 'involuntary_termination', 'voided']

🔹 Filtering by simplicity (frequency)...
Original word frequency: 3.75
Filtered simpler candidates:
suspended (freq=4.32)
withdrawn (freq=3.90)

🔹 Re-ranking with DistilBERT MLM...

Re-ranked candidates:
suspended | LM=0.0099 | Simplicity=4.32 | Final=1.3029
withdrawn | LM=0.0022 | Simplicity=3.90 | Final=1.1715

✅ Final Simplified Sentence:
The project was abruptly suspended.

====================================
Original sentence: She exhibited remarkable resilience.

🔹 Generating CBOW neighbors for: 'remarkable'
Top CBOW candidates:
['astonishing', 'astounding', 'amazing', 'incredible', 'extraordinary', 'impressive', 'phenomenal', 'marvelous', 'startling', 'stunning', 'miraculous', 'breathtaking', 'magnificent', 'noteworthy', 'stupendous', 'dramatic', 'surprising', 'splendid', 'heartening', 'admirable']

🔹 Filtering by simplicity (frequency)...
Original word frequency: 4.29
Filtered simpler candidates:
amazing (freq=5.10)
incredible (freq=4.63)
impressive (freq=4.47)
dramatic (freq=4.41)
extraordinary (freq=4.36)

🔹 Re-ranking with DistilBERT MLM...

Re-ranked candidates:
amazing | LM=0.0011 | Simplicity=5.10 | Final=1.5308
incredible | LM=0.0053 | Simplicity=4.63 | Final=1.3927
impressive | LM=0.0011 | Simplicity=4.47 | Final=1.3417
extraordinary | LM=0.0435 | Simplicity=4.36 | Final=1.3385
dramatic | LM=0.0016 | Simplicity=4.41 | Final=1.3241

✅ Final Simplified Sentence:
She exhibited amazing resilience.

====================================
Original sentence: The corporation initiated a comprehensive restructuring process.

🔹 Generating CBOW neighbors for: 'comprehensive'
Top CBOW candidates:
['Visit_www.teldta.com', 'acomprehensive', 'Comprehensive', 'thorough', 'Plesk_Virtuozzo_PEM_HSPcomplete', 'detailed', 'LoanPerformance_HPI_provides', 'fully_integrated', 'extensive', 'sealants_potting', 'holistic', 'integrated', 'StillSecure_delivers', 'prebuilt_analytic_applications', 'includes_##/##-bit_RISC', 'Manitex_subsidiary', 'Shire_velaglucerase_alfa', 'BBSI_provides', 'engineered_adhesives_coatings', 'broadest']

🔹 Filtering by simplicity (frequency)...
Original word frequency: 4.36
Filtered simpler candidates:
detailed (freq=4.51)
extensive (freq=4.49)

🔹 Re-ranking with DistilBERT MLM...

Re-ranked candidates:
detailed | LM=0.0031 | Simplicity=4.51 | Final=1.3552
extensive | LM=0.0010 | Simplicity=4.49 | Final=1.3477

✅ Final Simplified Sentence:
The corporation initiated a detailed restructuring process.

====================================
Original sentence: His intentions were ambiguous.

🔹 Generating CBOW neighbors for: 'His intentions were ambiguous.'
⚠ Word not in vocabulary
⚠ No CBOW candidates found

====================================
Original sentence: We must evacuate the premises immediately.

🔹 Generating CBOW neighbors for: 'evacuate'
Top CBOW candidates:
['evacuation', 'evacuated', 'evacuating', 'evacuations', 'mandatory_evacuation', 'voluntary_evacuation', 'mandatory_evacuations', 'flee', 'Evacuations', 'voluntary_evacuations', 'stay_indoors', 'Evacuation', 'ordered_mandatory_evacuations', 'evacuates', 'evacuted', 'mandatory_evacuation_order', 'Evacuating', 'evacuees', 'mandatory_evacuation_orders', 'Voluntary_evacuations']

🔹 Filtering by simplicity (frequency)...
Original word frequency: 3.53
Filtered simpler candidates:
flee (freq=3.81)
evacuation (freq=3.74)
Evacuation (freq=3.74)
evacuated (freq=3.66)

🔹 Re-ranking with DistilBERT MLM...

Re-ranked candidates:
flee | LM=0.0069 | Simplicity=3.81 | Final=1.1478
evacuation | LM=0.0000 | Simplicity=3.74 | Final=1.1220
evacuated | LM=0.0001 | Simplicity=3.66 | Final=1.0981

✅ Final Simplified Sentence:
We must flee the premises immediately.

====================================
Original sentence: We must evacuate the premises immediately.

🔹 Generating CBOW neighbors for: 'premises'
Top CBOW candidates:
['premesis', 'Linenhall_Street', 'shoplot', 'portacabin', 'portacabins', 'godown', 'SpectraLink_handsets_free', 'Kirkstall_Road', 'Jalan_Kebangsaan_Lama', 'shoplots', 'hostel', 'Nissen_Hut', 'Segaiya', 'louvre_blades', 'WifiCasino_GS_Concierge', 'Mutley_Plain', 'liquor_vend', 'Spar_shop', 'Thangal_bazar', 'Eastbank_Street']

🔹 Filtering by simplicity (frequency)...
Original word frequency: 4.02
Filtered simpler candidates:
⚠ No simpler candidates found

====================================
Original sentence: That explanation is completely absurd.

🔹 Generating CBOW neighbors for: 'absurd'
Top CBOW candidates:
['ludicrous', 'ridiculous', 'preposterous', 'laughable', 'nonsensical', 'illogical', 'outrageous', 'asinine', 'outlandish', 'utterly_ridiculous', 'implausible', 'idiotic', 'disingenuous', 'patently_absurd', 'silly', 'incomprehensible', 'fatuous', 'absurb', 'far_fetched', 'specious']

🔹 Filtering by simplicity (frequency)...
Original word frequency: 3.95
Filtered simpler candidates:
ridiculous (freq=4.46)
silly (freq=4.39)

🔹 Re-ranking with DistilBERT MLM...

Re-ranked candidates:
ridiculous | LM=0.0105 | Simplicity=4.46 | Final=1.3454
silly | LM=0.0005 | Simplicity=4.39 | Final=1.3174

✅ Final Simplified Sentence:
That explanation is completely ridiculous.

====================================
Original sentence: I cannot tolerate this outrageous behavior.

🔹 Generating CBOW neighbors for: 'tolerate'
Top CBOW candidates:
['tolerated', 'condone', 'tolerating', 'tolerates', 'condoning', 'tacitly_condone', 'condones', 'deplore', 'condoned', 'succumb', 'condemn', 'tolerant', 'abhor', 'Condoning', 'endure', 'zero_tolerance', 'deter', 'accept', 'suffer', 'Tolerating']

🔹 Filtering by simplicity (frequency)...
Original word frequency: 3.84
Filtered simpler candidates:
accept (freq=4.85)
suffer (freq=4.45)

🔹 Re-ranking with DistilBERT MLM...

Re-ranked candidates:
accept | LM=0.0121 | Simplicity=4.85 | Final=1.4635
suffer | LM=0.0002 | Simplicity=4.45 | Final=1.3351

✅ Final Simplified Sentence:
I cannot accept this outrageous behavior.

====================================
Original sentence: The scientist formulated a groundbreaking hypothesis.

🔹 Generating CBOW neighbors for: 'groundbreaking'
Top CBOW candidates:
['pioneering', 'Groundbreaking', 'tight_bodice_cinched', 'trailblazing', 'cutting_edge', 'paradigm_shifting', 'landmark', 'pathbreaking', 'innovative', 'transformative', 'No_Fences_Ropin', 'breakthrough', 'ceremonial_groundbreaking', 'unveiling', 'seminal', 'groundbreaking_ceremony', 'revolutionize', 'Pioneering', 'FASTforward_blog', 'revolutionary']

🔹 Filtering by simplicity (frequency)...
Original word frequency: 3.46
Filtered simpler candidates:
revolutionary (freq=4.17)
innovative (freq=4.10)
landmark (freq=3.89)
breakthrough (freq=3.83)
pioneering (freq=3.54)

🔹 Re-ranking with DistilBERT MLM...

Re-ranked candidates:
revolutionary | LM=0.0002 | Simplicity=4.17 | Final=1.2511
innovative | LM=0.0000 | Simplicity=4.10 | Final=1.2300
landmark | LM=0.0001 | Simplicity=3.89 | Final=1.1671
breakthrough | LM=0.0004 | Simplicity=3.83 | Final=1.1493
pioneering | LM=0.0001 | Simplicity=3.54 | Final=1.0621

✅ Final Simplified Sentence:
The scientist formulated a revolutionary hypothesis.

====================================
Original sentence: The scientist formulated a groundbreaking hypothesis.

🔹 Generating CBOW neighbors for: 'hypothesis'
Top CBOW candidates:
['hypotheses', 'theory', 'theories', 'supposition', 'postulate', 'amyloid_hypothesis', 'empirical', 'amyloid_cascade', 'prion_hypothesis', 'hypothesized', 'hypothesize', 'null_hypothesis', 'notion', 'testable_hypothesis', 'thesis', 'empirical_evidence', 'neo_Darwinism', 'anthropic_principle', 'Hypothesis', 'empirically']

🔹 Filtering by simplicity (frequency)...
Original word frequency: 3.87
Filtered simpler candidates:
theory (freq=4.88)
theories (freq=4.22)
notion (freq=4.17)
thesis (freq=4.05)

🔹 Re-ranking with DistilBERT MLM...

Re-ranked candidates:
theory | LM=0.1184 | Simplicity=4.88 | Final=1.5469
thesis | LM=0.0898 | Simplicity=4.05 | Final=1.2779
theories | LM=0.0008 | Simplicity=4.22 | Final=1.2666
notion | LM=0.0015 | Simplicity=4.17 | Final=1.2521

✅ Final Simplified Sentence:
The scientist formulated a groundbreaking theory.

====================================
Original sentence: The project was abruptly terminated due to financial constraints.

🔹 Generating CBOW neighbors for: 'abruptly'
Top CBOW candidates:
['abrupt', 'abrubtly', 'suddenly', 'Abruptly', 'unceremoniously', 'prematurely', 'unexpectedly', 'mysteriously', 'Roger_buh', 'altogether', 'temporarily', 'midsentence', 'inexplicably', 'Seidlin_teared', 'shortly_afterward', 'after', 'shortly_thereafter', 'MSNBC_simulcast', 'briefly', 'Stadnik_loves']

🔹 Filtering by simplicity (frequency)...
Original word frequency: 3.56
Filtered simpler candidates:
after (freq=6.11)
suddenly (freq=4.69)
briefly (freq=4.22)
altogether (freq=4.07)
temporarily (freq=4.06)

🔹 Re-ranking with DistilBERT MLM...

Re-ranked candidates:
after | LM=0.0000 | Simplicity=6.11 | Final=1.8330
suddenly | LM=0.0003 | Simplicity=4.69 | Final=1.4072
briefly | LM=0.0030 | Simplicity=4.22 | Final=1.2681
temporarily | LM=0.0554 | Simplicity=4.06 | Final=1.2567
altogether | LM=0.0000 | Simplicity=4.07 | Final=1.2210

✅ Final Simplified Sentence:
The project was after terminated due to financial constraints.

====================================
Original sentence: The committee unanimously endorsed the controversial legislation.

🔹 Generating CBOW neighbors for: 'endorsed'
Top CBOW candidates:
['endorsing', 'endorses', 'enthusiastically_endorsed', 'endorse', 'unanimously_endorsed', 'supported', 'advocated', 'championed', 'backed', 'heartily_endorsed', 'Endorsed', 'opposed', 'embraced', 'approved', 'espoused', 'endorsement', 'vehemently_opposed', 'opposes', 'rejected', 'Endorsing']

🔹 Filtering by simplicity (frequency)...
Original word frequency: 3.87
Filtered simpler candidates:
supported (freq=4.68)
approved (freq=4.64)
opposed (freq=4.51)
rejected (freq=4.37)
backed (freq=4.26)

🔹 Re-ranking with DistilBERT MLM...

Re-ranked candidates:
approved | LM=0.4677 | Simplicity=4.64 | Final=1.7194
rejected | LM=0.2796 | Simplicity=4.37 | Final=1.5067
supported | LM=0.0099 | Simplicity=4.68 | Final=1.4109
opposed | LM=0.0047 | Simplicity=4.51 | Final=1.3563
backed | LM=0.0013 | Simplicity=4.26 | Final=1.2789

✅ Final Simplified Sentence:
The committee unanimously approved the controversial legislation.

====================================
Original sentence: The committee unanimously endorsed the controversial legislation.

🔹 Generating CBOW neighbors for: 'legislation'
Top CBOW candidates:
['bill', 'Legislation', 'amendment', 'amendments', 'legistlation', 'legisation', 'S.####', 'S.###', 'omnibus_bill', 'repeal', 'Arbitration_Fairness_Act', 'laws', 'legisla_tion', 'constitutional_amendment', 'Senate', 'constitutional_amendment_HJR', 'HB####', 'Cybersecurity_Enhancement_Act', 'Stupak_Pitts_amendment', 'NICS_Improvement_Act']

🔹 Filtering by simplicity (frequency)...
Original word frequency: 4.55
Filtered simpler candidates:
S.#### (freq=5.86)
S.### (freq=5.86)
bill (freq=5.15)
laws (freq=4.89)
Senate (freq=4.80)

🔹 Re-ranking with DistilBERT MLM...

Re-ranked candidates:
bill | LM=0.0769 | Simplicity=5.15 | Final=1.5988
laws | LM=0.0006 | Simplicity=4.89 | Final=1.4674

✅ Final Simplified Sentence:
The committee unanimously endorsed the controversial bill.
###
