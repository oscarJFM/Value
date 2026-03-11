import os
import pandas as pd
import time
import subprocess  # <--- New Import
from flask import Flask, jsonify, render_template, request
from collections import defaultdict
from pathlib import Path

from update_inventory import execute_transfer

def get_ai_alternative_id(medicine_id):
    """Get the AI-suggested alternative medicine ID for a given medicine"""
    alternative_mapping = {
        # since inventory now uses only base strengths, map directly
        'M001': 'M002',  # Paracetamol <-> Ibuprofen
        'M002': 'M001',  # Ibuprofen <-> Paracetamol
        'M003': 'M013',  # Amoxicillin -> Cephalexin 500mg
        'M004': 'M014',  # Insulin -> Insulin Lispro
        'M005': 'M015',  # Epinephrine -> Norepinephrine
        'M006': 'M016',  # Salbutamol -> Terbutaline inhaler
        'M007': 'M017',  # Morphine -> Fentanyl patch
        'M008': 'M018',  # Aspirin -> Clopidogrel 75mg
        'M009': 'M019',  # Metformin -> Glipizide 5mg
        'M010': 'M020'   # Atorvastatin -> Simvastatin 20mg
    }
    return alternative_mapping.get(medicine_id)

def check_local_alternative_available(hospital_source, medicine_id, df_full_network):
    """Check if the hospital has the AI-suggested alternative available locally"""
    alternative_id = get_ai_alternative_id(medicine_id)
    if not alternative_id:
        return False
    
    # Check if this hospital has the alternative in stock
    local_alternative = df_full_network[
        (df_full_network['ID'] == alternative_id) & 
        (df_full_network['Hospital_Source'] == hospital_source) &
        (df_full_network['Amount'] > 0)
    ]
    
    return not local_alternative.empty

def LLM_advisory(medicine_id, medicine_name):
    """
    Simulated LLM function for clinical advisory when no H2H transfer is possible.
    This function will be replaced with actual LLM API call later.
    
    Args:
        medicine_id (str): The ID of the missing medicine
        medicine_name (str): The name of the missing medicine
    
    Returns:
        dict: Contains 'therapeutic_alternative' and 'clinical_rationale'
    """
    # Mapping original medicines to their therapeutic alternatives (now in inventory as M011-M020)
    advisory_database = {
        'M001': {  # Paracetamol
            'therapeutic_alternative': 'M002 - Ibuprofen',
            'clinical_rationale': 'Both are analgesics/antipyretics. Ibuprofen provides similar pain relief and fever reduction. Monitor for GI contraindications.'
        },
        'M002': {  # Ibuprofen
            'therapeutic_alternative': 'M001 - Paracetamol',
            'clinical_rationale': 'Alternative analgesic with different mechanism. Safer for patients with GI issues or cardiovascular risk factors.'
        },
        'M003': {  # Amoxicillin
            'therapeutic_alternative': 'M013 - Cephalexin 500mg',
            'clinical_rationale': 'First-generation cephalosporin with similar spectrum. Effective against gram-positive bacteria. Check penicillin allergy history.'
        },
        'M004': {  # Insulin
            'therapeutic_alternative': 'M014 - Insulin Lispro',
            'clinical_rationale': 'CRITICAL: Insulin substitution requires careful dosing adjustment. Consult endocrinologist immediately for conversion protocols.'
        },
        'M005': {  # Epinephrine
            'therapeutic_alternative': 'M015 - Norepinephrine',
            'clinical_rationale': 'EMERGENCY: For anaphylaxis, no substitute exists. For cardiac support, norepinephrine may be considered with dose adjustment.'
        },
        'M006': {  # Salbutamol
            'therapeutic_alternative': 'M016 - Terbutaline inhaler',
            'clinical_rationale': 'Alternative beta-2 agonist bronchodilator. Similar efficacy for acute bronchospasm. Adjust dosing per protocol.'
        },
        'M007': {  # Morphine
            'therapeutic_alternative': 'M017 - Fentanyl patch',
            'clinical_rationale': 'Equianalgesic opioid conversion required. Fentanyl: 1mg morphine = 0.01mg fentanyl. Monitor respiratory status closely.'
        },
        'M008': {  # Aspirin
            'therapeutic_alternative': 'M018 - Clopidogrel 75mg',
            'clinical_rationale': 'Alternative antiplatelet agent. Different mechanism but similar cardiovascular protection. Monitor bleeding risk.'
        },
        'M009': {  # Metformin
            'therapeutic_alternative': 'M019 - Glipizide 5mg',
            'clinical_rationale': 'Different class (sulfonylurea vs biguanide). Monitor for hypoglycemia risk. Consider insulin if severe diabetes.'
        },
        'M010': {  # Atorvastatin
            'therapeutic_alternative': 'M020 - Simvastatin 20mg',
            'clinical_rationale': 'Alternative HMG-CoA reductase inhibitor. Similar efficacy for cholesterol management. Monitor liver function.'
        },

        'M017': {  # Fentanyl patch
            'therapeutic_alternative': 'Consult Clinical Pharmacist',
            'clinical_rationale': 'No standard alternative for fentanyl patch. Immediate consultation required for pain management options.'
        }
    }
    
    # Return specific advisory or generic fallback
    if medicine_id in advisory_database:
        return advisory_database[medicine_id]
    else:
        return {
            'therapeutic_alternative': 'Consult Clinical Pharmacist',
            'clinical_rationale': f'No standard alternative identified for {medicine_name}. Immediate pharmacist consultation required for therapeutic substitution.'
        }

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent / "medicine_inventory_dummy_data_v2"

def get_data():
    # 1. RUN THE INVENTORY SCRIPT AUTOMATICALLY
    # This runs: python manage_inventory.py --base-dir "Value/medicine_inventory_dummy_data_v2"
    try:
        subprocess.run(
            [
                "python",
                "manage_inventory.py",
                "--base-dir",
                str(BASE_DIR),
            ],
            check=True,
        )
        print("Inventory script synced successfully.")
    except Exception as e:
        print(f"Error running inventory script: {e}")

    # 2. NOW READ THE DATA (Which is now fresh)
    base_path = BASE_DIR
    hosp_a = pd.read_csv(base_path / 'Hospital_A_inventory.csv')
    
    network_list = []
    for letter in ['B', 'C', 'D', 'E']:
        path = base_path / f'Hospital_{letter}_inventory.csv'
        if path.exists():
            df = pd.read_csv(path)
            df['Hospital_Source'] = f'Hospital {letter}'
            network_list.append(df)
    
    network_df = pd.concat(network_list, ignore_index=True) if network_list else pd.DataFrame()
    return hosp_a, network_df

@app.route('/')
def index():
    df_a, df_net = get_data()
    df_full_network = pd.concat([df_a.assign(Hospital_Source='Hospital A'), df_net], ignore_index=True)
    
    # Identify items needing rescue (< 5 units)
    all_shortages = df_full_network[df_full_network['Amount'] < 5].to_dict('records') 
    
    grouped_results = defaultdict(list)
    hospitals_in_incident = set()

    # Track last sync for the footer
    file_path = BASE_DIR / 'Hospital_A_inventory.csv'
    last_sync = time.ctime(os.path.getmtime(file_path))

    for item in all_shortages:
        # Data Cleaning: Convert CSV strings to integers
        urgency_val = int(item.get('Urgency', 1))
        amount = int(item.get('Amount', 0))
        
        # Check if this shortage can be resolved by local AI alternative
        if amount == 0 and check_local_alternative_available(item['Hospital_Source'], item['ID'], df_full_network):
            # Skip this shortage - it can be resolved locally with AI alternative
            continue
        
        item['status_label'] = "STOCKOUT" if amount == 0 else "LOW STOCK"
        hospitals_in_incident.add(item['Hospital_Source'])
        
        # Initialize solution fields
        item['solution'] = None
        item['ai_suggestion'] = None
        item['ai_rationale'] = None
        item['alt_id'] = None
        
        # FIND SOLUTION
        potential_lenders = df_full_network[
            (df_full_network['ID'] == item['ID']) & 
            (df_full_network['Hospital_Source'] != item['Hospital_Source']) &
            (df_full_network['Amount'] > 0)
        ]
        
        if not potential_lenders.empty:
            # H2H Transfer Available
            best_match = potential_lenders.loc[potential_lenders['Amount'].idxmax()]
            item['solution'] = {'Facility': best_match['Hospital_Source'], 'Qty': best_match['Amount'], 'Distance': 12.5}
        else:
            # No H2H available - Check if AI alternative exists in network
            alternative_id = get_ai_alternative_id(item['ID'])
            if alternative_id:
                # Check if ANY hospital has the alternative
                network_alternatives = df_full_network[
                    (df_full_network['ID'] == alternative_id) & 
                    (df_full_network['Amount'] > 0)
                ]
                
                if not network_alternatives.empty:
                    total_alt_amount = network_alternatives['Amount'].sum()
                    if total_alt_amount >= 5:
                        # AI alternative available in sufficient quantity
                        item['solution'] = None
                        item['ai_suggestion'] = f'{alternative_id} - {network_alternatives.iloc[0]["Medicine"]}'
                        item['ai_rationale'] = 'Alternative available in network'
                        item['alt_id'] = alternative_id
                    else:
                        # Alternative available but low - show LLM advisory for the alternative
                        item['solution'] = None
                        alt_medicine = network_alternatives.iloc[0]['Medicine']
                        llm_response = LLM_advisory(alternative_id, alt_medicine)
                        item['ai_suggestion'] = llm_response['therapeutic_alternative']
                        item['ai_rationale'] = llm_response['clinical_rationale']
                        item['alt_id'] = alternative_id
                else:
                    # No alternative available anywhere - critical shortage
                    item['solution'] = None
                    item['ai_suggestion'] = 'CRITICAL: No alternatives available in network'
                    item['ai_rationale'] = f'Total network shortage of {item["Medicine"]} and its therapeutic alternative. Emergency procurement required.'
                    item['alt_id'] = None
        # RISK CALCULATION (1-10 Scale)
        score = float(urgency_val)
        score += 3.0 if amount == 0 else 1.0
        
        # High-danger scoring for LLM dependency (total network stockout)
        if not item['solution']:
            score = 9.5  # Automatically set to high-danger level for LLM cases
            
        item['risk_score'] = min(score, 10.0)
        
        # Color Logic for HTML - Changed Info (Blue) to Success (Green)
        if item['risk_score'] >= 8.0:
            item['risk_class'] = 'bg-danger' # Red
        elif item['risk_score'] >= 5.0:
            item['risk_class'] = 'bg-warning text-dark' # Yellow
        else:
            item['risk_class'] = 'bg-success' # NOW GREEN

        grouped_results[item['Hospital_Source']].append(item)

    # Sort each hospital group by risk score (highest first)
    for hosp in grouped_results:
        grouped_results[hosp] = sorted(grouped_results[hosp], key=lambda x: x['risk_score'], reverse=True)

    header_text = ", ".join(sorted(hospitals_in_incident)) if hospitals_in_incident else "No Active Incidents"

    return render_template('index.html', 
                           grouped_shortages=grouped_results, 
                           incident_hospitals=header_text,
                           last_sync=last_sync)


@app.route('/get_alternative_sources', methods=['POST'])
def get_alternative_sources():
    """Get hospitals that have the AI-suggested alternative medicine in stock"""
    payload = request.get_json(force=True, silent=True) or {}
    try:
        alternative_id = payload['alternative_id']
    except KeyError:
        return jsonify({'status': 'error', 'message': 'Missing alternative_id'}), 400

    try:
        # Get fresh data
        df_a, df_net = get_data()
        df_full_network = pd.concat([df_a.assign(Hospital_Source='Hospital A'), df_net], ignore_index=True)
        
        # Find hospitals with the alternative medicine in stock
        available_sources = df_full_network[
            (df_full_network['ID'] == alternative_id) & 
            (df_full_network['Amount'] > 0)
        ]
        
        sources = []
        for _, row in available_sources.iterrows():
            sources.append({
                'hospital': row['Hospital_Source'],
                'amount': int(row['Amount']),
                'medicine': row['Medicine']
            })
        
        return jsonify({
            'status': 'ok',
            'sources': sources,
            'alternative_id': alternative_id
        })
        
    except Exception as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 500


@app.route('/transfer', methods=['POST'])
def transfer():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        medicine_id = payload['medicine_id']
        lender = payload['lender']
        receiver = payload['receiver']
        amount = int(payload['amount'])
    except (KeyError, TypeError, ValueError):
        return jsonify({'status': 'error', 'message': 'Invalid transfer payload'}), 400

    try:
        result = execute_transfer(BASE_DIR, medicine_id, lender, receiver, amount)
    except Exception as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 400

    return jsonify(
        {
            'status': 'ok',
            'message': (
                f"Transferred {result.quantity} units of {result.medicine_name} "
                f"from {result.lender} to {result.receiver}."
            ),
            'batches': result.transferred_batches,
        }
    )

if __name__ == '__main__':
    app.run(debug=True)