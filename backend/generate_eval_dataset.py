import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

QUESTIONS = [
    # --- 10 SIMPLE (Factual / Single Chunk) ---
    {
        "question": "What imaging is recommended for chronic chest pain?",
        "correct_answer": "Coronary CT Angiography (CCTA) or stress testing are typically recommended for chronic chest pain depending on clinical probability."
    },
    {
        "question": "What does aspirin treat?",
        "correct_answer": "Aspirin is used as an antiplatelet agent to prevent thrombosis and reduce the risk of cardiovascular events like myocardial infarction."
    },
    {
        "question": "What is the primary use of warfarin?",
        "correct_answer": "Warfarin is an anticoagulant used to prevent and treat blood clots, commonly for atrial fibrillation or deep vein thrombosis."
    },
    {
        "question": "What is a common symptom of acute myocardial infarction?",
        "correct_answer": "Severe, crushing chest pain radiating to the jaw or left arm, often accompanied by shortness of breath and diaphoresis."
    },
    {
        "question": "What does an elevated troponin level indicate?",
        "correct_answer": "Elevated troponin generally indicates myocardial injury or infarction."
    },
    {
        "question": "Which medication is typically prescribed as a first-line treatment for hypertension?",
        "correct_answer": "ACE inhibitors, ARBs, calcium channel blockers, or thiazide diuretics are typically prescribed."
    },
    {
        "question": "What is the diagnostic purpose of an ECG?",
        "correct_answer": "An ECG is used to measure the electrical activity of the heart to detect arrhythmias, ischemia, or infarction."
    },
    {
        "question": "What condition is characterized by chronic high blood sugar?",
        "correct_answer": "Diabetes mellitus."
    },
    {
        "question": "Name a common contraindication for NSAIDs.",
        "correct_answer": "Chronic kidney disease, active peptic ulcer disease, or severe heart failure are common contraindications."
    },
    {
        "question": "What is a D-dimer test used for?",
        "correct_answer": "It is used to help rule out the presence of a serious blood clot, such as deep vein thrombosis or pulmonary embolism."
    },

    # --- 10 MODERATE (2 Related Chunks) ---
    {
        "question": "What diagnostic test is recommended for chest pain with suspected coronary disease?",
        "correct_answer": "An initial ECG followed by Coronary CT Angiography or functional stress testing is recommended."
    },
    {
        "question": "Which drug interactions exist for warfarin?",
        "correct_answer": "Warfarin interacts with many drugs including amiodarone, NSAIDs, certain antibiotics, and SSRIs, increasing bleeding risk."
    },
    {
        "question": "How should a patient with asthma and hypertension be treated?",
        "correct_answer": "Cardioselective beta-blockers, ACE inhibitors, or ARBs are preferred, avoiding nonselective beta-blockers which can trigger bronchospasm."
    },
    {
        "question": "What precautions should be taken when prescribing metformin to a patient undergoing a CT scan with contrast?",
        "correct_answer": "Metformin should be temporarily discontinued at the time of or before the procedure due to the risk of contrast-induced nephropathy and subsequent lactic acidosis."
    },
    {
        "question": "Why is troponin monitoring important when prescribing certain chemotherapy drugs?",
        "correct_answer": "Certain chemotherapies are cardiotoxic and can cause myocardial injury, which is monitored by elevated troponin levels."
    },
    {
        "question": "What are the risks of using ACE inhibitors in patients with severe kidney disease?",
        "correct_answer": "While protective in early disease, they can cause hyperkalemia and an acute decline in glomerular filtration rate in advanced severe kidney disease."
    },
    {
        "question": "Which imaging modality is best for detecting a suspected acute ischemic stroke?",
        "correct_answer": "Non-contrast CT head is the initial test to rule out hemorrhage, followed by MRI of the brain for definitive detection of ischemia."
    },
    {
        "question": "How do statins interact with macrolide antibiotics?",
        "correct_answer": "Macrolides inhibit the metabolism of certain statins (like atorvastatin and simvastatin), significantly increasing the risk of myopathy and rhabdomyolysis."
    },
    {
        "question": "What is the relationship between chronic kidney disease and anemia?",
        "correct_answer": "Kidneys produce erythropoietin; chronic kidney disease reduces this production, leading to decreased red blood cell synthesis and anemia."
    },
    {
        "question": "What alternative anticoagulant is recommended if a patient develops heparin-induced thrombocytopenia (HIT)?",
        "correct_answer": "A direct thrombin inhibitor, such as argatroban or bivalirudin, or fondaparinux is recommended."
    },

    # --- 20 COMPLEX MULTI-HOP (Symptoms + Diseases + Drugs + Contraindications) ---
    {
        "question": "Patient has chest pain, diabetes, and kidney issues. What possible conditions, tests, and treatment precautions should be considered?",
        "correct_answer": "Conditions include atypical acute coronary syndrome due to diabetes. Tests: ECG and troponin. Precautions: Avoid nephrotoxic contrast during imaging (CT angiography) due to kidney issues, and adjust renally cleared medications."
    },
    {
        "question": "A patient with glioblastoma is undergoing MRI evaluation and CAR-T therapy. What diagnostic improvements and treatment relationships exist?",
        "correct_answer": "Advanced MRI helps differentiate true tumor progression from pseudoprogression caused by immune therapies like CAR-T. Treatment requires careful monitoring of neurotoxicity."
    },
    {
        "question": "A patient taking warfarin reports chest pain and has kidney disease. What diagnostic tests and medication risks should be considered?",
        "correct_answer": "Tests: ECG, troponin, and non-contrast CT. Risks: Warfarin increases bleeding risk if invasive procedures are needed for chest pain (like PCI). Kidney disease alters drug clearance, increasing the risk of warfarin toxicity."
    },
    {
        "question": "If a patient has respiratory symptoms and is allergic to specific antibiotics (like penicillin), what alternative treatment considerations exist?",
        "correct_answer": "For bacterial respiratory infections, macrolides or respiratory fluoroquinolones may be considered, but prescribers must verify cross-reactivity and monitor for prolonged QT intervals with these alternatives."
    },
    {
        "question": "A diabetic patient with hypertension and chronic chest pain needs imaging. What imaging and medication risks should be evaluated?",
        "correct_answer": "If considering CT with contrast, assess renal function to prevent contrast-induced nephropathy. Metformin must be held. Beta-blockers may mask hypoglycemia symptoms in diabetics."
    },
    {
        "question": "A patient presents with acute shortness of breath, a history of deep vein thrombosis, and is taking a direct oral anticoagulant (DOAC). What conditions and diagnostic pathways are recommended?",
        "correct_answer": "Suspect pulmonary embolism despite DOAC therapy. A CT pulmonary angiogram or V/Q scan is recommended. Evaluate for medication non-adherence or DOAC failure."
    },
    {
        "question": "A patient with chronic liver disease and atrial fibrillation needs anticoagulation. What are the drug safety risks and recommended monitoring?",
        "correct_answer": "Liver disease alters coagulation and drug metabolism. Warfarin is difficult to dose due to fluctuating INR, and DOACs require careful dosage adjustment or are contraindicated depending on liver function severity."
    },
    {
        "question": "What are the diagnostic and treatment considerations for a patient with new-onset heart failure, concurrent COPD, and a severe allergy to ACE inhibitors?",
        "correct_answer": "Echocardiogram and BNP for diagnosis. For treatment, substitute ACE inhibitors with an ARB or ARNI. Use cardioselective beta-blockers to avoid exacerbating COPD."
    },
    {
        "question": "A rheumatoid arthritis patient on chronic NSAIDs and methotrexate presents with melena and severe fatigue. What diagnostic tests and treatment adjustments are urgent?",
        "correct_answer": "Suspect upper GI bleed from NSAIDs. Urgent endoscopy, CBC to check for anemia, and metabolic panel. Stop NSAIDs immediately and evaluate methotrexate toxicity."
    },
    {
        "question": "An elderly patient with Parkinson's disease and hypertension presents with recurrent syncope. What drug interactions and diagnostic tests should be investigated?",
        "correct_answer": "Investigate orthostatic hypotension caused by the interaction of Parkinson's medications (e.g., levodopa) and antihypertensives. Perform orthostatic vitals, ECG, and review medication timing."
    },
    {
        "question": "A patient with severe renal impairment is diagnosed with a life-threatening systemic fungal infection. What treatment and monitoring complications arise?",
        "correct_answer": "Standard antifungals like amphotericin B are highly nephrotoxic. Liposomal formulations or alternative antifungals requiring renal dose adjustment must be used, with daily monitoring of creatinine and electrolytes."
    },
    {
        "question": "A pregnant patient with a history of mechanical heart valve requires anticoagulation. What are the medication risks across different trimesters?",
        "correct_answer": "Warfarin crosses the placenta and is teratogenic in the first trimester, requiring a switch to low-molecular-weight heparin (LMWH). Warfarin may be resumed in the second trimester but must be switched back to LMWH near delivery to prevent fetal bleeding."
    },
    {
        "question": "A patient with bipolar disorder taking lithium presents with polyuria and elevated creatinine. What disease mechanism and diagnostic steps are needed?",
        "correct_answer": "Suspect lithium-induced nephrogenic diabetes insipidus or chronic kidney disease. Check lithium levels, serum electrolytes, urine osmolality, and consider transitioning to alternative mood stabilizers."
    },
    {
        "question": "A patient with a history of gastric bypass surgery presents with peripheral neuropathy and macrocytic anemia. What absorption pathway is affected and what is the treatment?",
        "correct_answer": "Gastric bypass reduces intrinsic factor production and stomach acid, severely impairing Vitamin B12 absorption. Lifelong intramuscular or high-dose oral B12 supplementation is required."
    },
    {
        "question": "A patient taking an SSRI for depression is prescribed tramadol for acute back pain and develops a fever, tachycardia, and hyperreflexia. What is the condition and immediate response?",
        "correct_answer": "Suspect Serotonin Syndrome due to the interaction between the SSRI and tramadol. Immediately discontinue both offending agents and provide supportive care."
    },
    {
        "question": "A patient with active tuberculosis and HIV is starting antiretroviral therapy (ART). What drug interaction challenges exist between TB medications and ART?",
        "correct_answer": "Rifampin, a core TB drug, is a potent CYP450 inducer that significantly reduces the blood levels of many antiretrovirals (like protease inhibitors), requiring careful substitution (e.g., rifabutin) or ART dose adjustments."
    },
    {
        "question": "A patient with heart failure and gout is experiencing an acute gout flare. Why is treatment complex, and what are the alternatives?",
        "correct_answer": "NSAIDs are contraindicated as they worsen heart failure. Colchicine must be used carefully if renal function is reduced. Glucocorticoids are an alternative but may cause fluid retention exacerbating heart failure."
    },
    {
        "question": "A patient with liver cirrhosis and ascites develops confusion and altered mental status. What diagnostic testing and treatment protocol should be initiated?",
        "correct_answer": "Suspect hepatic encephalopathy. Check ammonia levels and rule out infection via diagnostic paracentesis (to exclude spontaneous bacterial peritonitis). Initiate lactulose and rifaximin."
    },
    {
        "question": "A patient on chronic glucocorticoids for lupus presents with hip pain. What imaging is necessary and what condition must be ruled out?",
        "correct_answer": "Suspect avascular necrosis of the femoral head due to long-term steroid use. An MRI of the hip is the most sensitive diagnostic imaging modality required."
    },
    {
        "question": "A patient with chronic stable angina is using sublingual nitroglycerin but is newly prescribed a PDE5 inhibitor for erectile dysfunction. What is the critical risk?",
        "correct_answer": "Concomitant use of nitroglycerin and PDE5 inhibitors causes profound, life-threatening hypotension. They are strictly contraindicated together."
    }
]

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_file = DATA_DIR / "ground_truth_questions.json"
    
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(QUESTIONS, f, indent=2)
    
    print(f"Successfully generated {len(QUESTIONS)} questions in {out_file.name}")
    print("Distribution: 10 Simple, 10 Moderate, 20 Complex")

if __name__ == "__main__":
    main()
