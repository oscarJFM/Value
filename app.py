import os
import pandas as pd
import time
import subprocess  # <--- New Import
from functools import wraps
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from collections import defaultdict
from pathlib import Path

from update_inventory import execute_transfer
from shortage_forecast import predict_upcoming_shortages


# ── LLM Alternative-Medicine Helpers ──────────────────────────────────────

def get_ai_alternative_id(medicine_id):
    """Get the AI-suggested alternative medicine ID for a given medicine"""
    alternative_mapping = {
        'M001': 'M002',   # Paracetamol <-> Ibuprofen
        'M002': 'M001',   # Ibuprofen <-> Paracetamol
        'M003': 'M013',   # Amoxicillin -> Cephalexin 500mg
        'M004': 'M014',   # Insulin -> Insulin Lispro
        'M005': 'M015',   # Epinephrine -> Norepinephrine
        'M006': 'M016',   # Salbutamol -> Terbutaline inhaler
        'M007': 'M017',   # Morphine -> Fentanyl patch
        'M008': 'M018',   # Aspirin -> Clopidogrel 75mg
        'M009': 'M019',   # Metformin -> Glipizide 5mg
        'M010': 'M020',   # Atorvastatin -> Simvastatin 20mg
    }
    return alternative_mapping.get(medicine_id)


def check_local_alternative_available(hospital_source, medicine_id, df_full_network):
    """Check if the hospital has the AI-suggested alternative available locally"""
    alternative_id = get_ai_alternative_id(medicine_id)
    if not alternative_id:
        return False
    local_alternative = df_full_network[
        (df_full_network['ID'] == alternative_id) &
        (df_full_network['Hospital_Source'] == hospital_source) &
        (df_full_network['Amount'] > 0)
    ]
    return not local_alternative.empty


def LLM_advisory(medicine_id, medicine_name):
    """
    Simulated LLM function for clinical advisory when no H2H transfer is possible.
    Returns dict with 'therapeutic_alternative' and 'clinical_rationale'.
    """
    advisory_database = {
        'M001': {
            'therapeutic_alternative': 'M002 - Ibuprofen',
            'clinical_rationale': 'Both are analgesics/antipyretics. Ibuprofen provides similar pain relief and fever reduction. Monitor for GI contraindications.'
        },
        'M002': {
            'therapeutic_alternative': 'M001 - Paracetamol',
            'clinical_rationale': 'Alternative analgesic with different mechanism. Safer for patients with GI issues or cardiovascular risk factors.'
        },
        'M003': {
            'therapeutic_alternative': 'M013 - Cephalexin 500mg',
            'clinical_rationale': 'First-generation cephalosporin with similar spectrum. Effective against gram-positive bacteria. Check penicillin allergy history.'
        },
        'M004': {
            'therapeutic_alternative': 'M014 - Insulin Lispro',
            'clinical_rationale': 'CRITICAL: Insulin substitution requires careful dosing adjustment. Consult endocrinologist immediately for conversion protocols.'
        },
        'M005': {
            'therapeutic_alternative': 'M015 - Norepinephrine',
            'clinical_rationale': 'EMERGENCY: For anaphylaxis, no substitute exists. For cardiac support, norepinephrine may be considered with dose adjustment.'
        },
        'M006': {
            'therapeutic_alternative': 'M016 - Terbutaline Inhaler',
            'clinical_rationale': 'Alternative beta-2 agonist bronchodilator. Similar efficacy for acute bronchospasm. Adjust dosing per protocol.'
        },
        'M007': {
            'therapeutic_alternative': 'M017 - Fentanyl Patch',
            'clinical_rationale': 'Equianalgesic opioid conversion required. Fentanyl: 1mg morphine = 0.01mg fentanyl. Monitor respiratory status closely.'
        },
        'M008': {
            'therapeutic_alternative': 'M018 - Clopidogrel 75mg',
            'clinical_rationale': 'Alternative antiplatelet agent. Different mechanism but similar cardiovascular protection. Monitor bleeding risk.'
        },
        'M009': {
            'therapeutic_alternative': 'M019 - Glipizide 5mg',
            'clinical_rationale': 'Different class (sulfonylurea vs biguanide). Monitor for hypoglycemia risk. Consider insulin if severe diabetes.'
        },
        'M010': {
            'therapeutic_alternative': 'M020 - Simvastatin 20mg',
            'clinical_rationale': 'Alternative HMG-CoA reductase inhibitor. Similar efficacy for cholesterol management. Monitor liver function.'
        },
    }
    if medicine_id in advisory_database:
        return advisory_database[medicine_id]
    return {
        'therapeutic_alternative': 'Consult Clinical Pharmacist',
        'clinical_rationale': f'No standard alternative identified for {medicine_name}. Immediate pharmacist consultation required.'
    }


def build_alternative_advice(hospital_source, medicine_id, medicine_name, df_full_network):
    """
    Build full alternative-medicine advice dict for a shortage item.
    Returns a dict with ai_suggestion, ai_rationale, alt_id, and best_source info,
    or None if no alternative mapping exists.
    """
    alternative_id = get_ai_alternative_id(medicine_id)
    if not alternative_id:
        return None

    llm = LLM_advisory(medicine_id, medicine_name)

    # Find all hospitals with the alternative in stock
    network_alt = df_full_network[
        (df_full_network['ID'] == alternative_id) &
        (df_full_network['Amount'] > 0)
    ]

    best_source = None
    if not network_alt.empty:
        best_row = network_alt.loc[network_alt['Amount'].idxmax()]
        best_source = {
            'Facility': best_row['Hospital_Source'],
            'Qty': int(best_row['Amount']),
            'Medicine': best_row['Medicine'],
        }

    return {
        'ai_suggestion': llm['therapeutic_alternative'],
        'ai_rationale': llm['clinical_rationale'],
        'alt_id': alternative_id,
        'alt_source': best_source,
    }


app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'nhs-shortage-rescue-dev-key-2026')

BASE_DIR = Path(__file__).resolve().parent / "medicine_inventory_dummy_data_v2"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "shortage_model.joblib"

# ── Dummy Login Credentials ──────────────────────────────────────────────
AUTHORISED_USERS = {
    'admin': 'admin123',
    'nurse': 'nurse123',
    'pharmacist': 'pharma123',
}


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('index'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if username in AUTHORISED_USERS and AUTHORISED_USERS[username] == password:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        error = 'Invalid username or password.'
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

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
@login_required
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

        # Skip shortages resolved by a local AI alternative
        if amount == 0 and check_local_alternative_available(item['Hospital_Source'], item['ID'], df_full_network):
            continue

        item['status_label'] = "STOCKOUT" if amount == 0 else "LOW STOCK"
        hospitals_in_incident.add(item['Hospital_Source'])

        # Initialize solution fields
        item['solution'] = None
        item['ai_suggestion'] = None
        item['ai_rationale'] = None
        item['alt_id'] = None
        item['alt_source'] = None

        # FIND SOLUTION - H2H first
        potential_lenders = df_full_network[
            (df_full_network['ID'] == item['ID']) & 
            (df_full_network['Hospital_Source'] != item['Hospital_Source']) &
            (df_full_network['Amount'] > 0)
        ]

        if not potential_lenders.empty:
            best_match = potential_lenders.loc[potential_lenders['Amount'].idxmax()]
            item['solution'] = {'Facility': best_match['Hospital_Source'], 'Qty': best_match['Amount'], 'Distance': 12.5}

        # Always attach AI alternative advice (shown alongside H2H or alone)
        alt_advice = build_alternative_advice(item['Hospital_Source'], item['ID'], item['Medicine'], df_full_network)
        if alt_advice:
            item['ai_suggestion'] = alt_advice['ai_suggestion']
            item['ai_rationale'] = alt_advice['ai_rationale']
            item['alt_id'] = alt_advice['alt_id']
            item['alt_source'] = alt_advice['alt_source']

        # RISK CALCULATION (1-10 Scale)
        score = float(urgency_val)
        score += 3.0 if amount == 0 else 1.0
        if not item['solution']:
            score = 9.5  # High-danger for LLM-only cases

        item['risk_score'] = min(score, 10.0)

        # Color Logic for HTML
        if item['risk_score'] >= 8.0:
            item['risk_class'] = 'bg-danger'
        elif item['risk_score'] >= 5.0:
            item['risk_class'] = 'bg-warning text-dark'
        else:
            item['risk_class'] = 'bg-success'

        grouped_results[item['Hospital_Source']].append(item)

    # Sort each hospital group by risk score (highest first)
    for hosp in grouped_results:
        grouped_results[hosp] = sorted(grouped_results[hosp], key=lambda x: x['risk_score'], reverse=True)

    header_text = ", ".join(sorted(hospitals_in_incident)) if hospitals_in_incident else "No Active Incidents"

    # Compute dashboard stat card values
    all_items = [item for items in grouped_results.values() for item in items]
    total_incidents = len(all_items)
    hospital_count = len(grouped_results)
    critical_count = sum(1 for item in all_items if item['risk_score'] >= 8.0)
    resolved_count = sum(1 for item in all_items if item['solution'])

    return render_template('index.html',
                           grouped_shortages=grouped_results,
                           incident_hospitals=header_text,
                           last_sync=last_sync,
                           total_incidents=total_incidents,
                           hospital_count=hospital_count,
                           critical_count=critical_count,
                           resolved_count=resolved_count)


@app.route('/forecasts')
@login_required
def forecasts():
    df_a, df_net = get_data()
    df_full_network = pd.concat([df_a.assign(Hospital_Source='Hospital A'), df_net], ignore_index=True)

    file_path = BASE_DIR / 'Hospital_A_inventory.csv'
    last_sync = time.ctime(os.path.getmtime(file_path))

    predicted_shortages = []
    try:
        forecast_rows = predict_upcoming_shortages(
            BASE_DIR,
            MODEL_PATH,
            probability_threshold=0.8,
        )
    except Exception as exc:
        print(f"Forecasting failed: {exc}")
        forecast_rows = []

    for row in forecast_rows:
        probability = float(row['probability'])
        hospital = row['Hospital_Source']
        urgency_val = float(row.get('urgency', 1))
        amount = int(row['current_amount'])

        risk_score = urgency_val
        risk_score += 3.0 if amount == 0 else 1.0

        forecast_item = {
            'Hospital_Source': hospital,
            'Medicine': row['Medicine'],
            'ID': row['Medicine_ID'],
            'Amount': int(row['current_amount']),
            'forecast_week': row['Week_Start_Date'],
            'forecast_window_weeks': row['forecast_window_weeks'],
            'probability': probability,
            'status_label': 'FORECASTED SHORTAGE',
            'solution': None,
            'ai_suggestion': None,
            'ai_rationale': None,
            'alt_id': None,
            'alt_source': None,
        }

        potential_lenders = df_full_network[
            (df_full_network['ID'] == row['Medicine_ID']) &
            (df_full_network['Hospital_Source'] != hospital) &
            (df_full_network['Amount'] > 0)
        ]

        if not potential_lenders.empty:
            best_match = potential_lenders.loc[potential_lenders['Amount'].idxmax()]
            forecast_item['solution'] = {
                'Facility': best_match['Hospital_Source'],
                'Qty': int(best_match['Amount'])
            }

        # Attach AI alternative advice
        alt_advice = build_alternative_advice(hospital, row['Medicine_ID'], row['Medicine'], df_full_network)
        if alt_advice:
            forecast_item['ai_suggestion'] = alt_advice['ai_suggestion']
            forecast_item['ai_rationale'] = alt_advice['ai_rationale']
            forecast_item['alt_id'] = alt_advice['alt_id']
            forecast_item['alt_source'] = alt_advice['alt_source']

        if not forecast_item['solution']:
            risk_score += 2.0

        forecast_item['risk_score'] = min(risk_score, 10.0)

        if forecast_item['risk_score'] >= 8.0:
            risk_class = 'bg-danger'
        elif forecast_item['risk_score'] >= 5.0:
            risk_class = 'bg-warning text-dark'
        else:
            risk_class = 'bg-success'

        forecast_item['risk_class'] = risk_class

        predicted_shortages.append(forecast_item)

    predicted_shortages = sorted(predicted_shortages, key=lambda x: x['probability'], reverse=True)[:10]

    grouped_forecasts = defaultdict(list)
    for item in predicted_shortages:
        grouped_forecasts[item['Hospital_Source']].append(item)

    for hosp in grouped_forecasts:
        grouped_forecasts[hosp] = sorted(grouped_forecasts[hosp], key=lambda x: x['risk_score'], reverse=True)

    # Compute forecast stat card values
    all_forecasts = [f for fs in grouped_forecasts.values() for f in fs]
    total_forecasts = len(all_forecasts)
    forecast_hospital_count = len(grouped_forecasts)
    forecast_critical_count = sum(1 for f in all_forecasts if f['risk_score'] >= 8.0)
    forecast_resolved_count = sum(1 for f in all_forecasts if f['solution'])

    return render_template('forecasts.html',
                           predicted_forecasts=grouped_forecasts,
                           last_sync=last_sync,
                           total_forecasts=total_forecasts,
                           forecast_hospital_count=forecast_hospital_count,
                           forecast_critical_count=forecast_critical_count,
                           forecast_resolved_count=forecast_resolved_count)


@app.route('/get_alternative_sources', methods=['POST'])
@login_required
def get_alternative_sources():
    """Get hospitals that have the AI-suggested alternative medicine in stock"""
    payload = request.get_json(force=True, silent=True) or {}
    try:
        alternative_id = payload['alternative_id']
    except KeyError:
        return jsonify({'status': 'error', 'message': 'Missing alternative_id'}), 400

    try:
        df_a, df_net = get_data()
        df_full_network = pd.concat([df_a.assign(Hospital_Source='Hospital A'), df_net], ignore_index=True)

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
@login_required
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
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)