import os
import pandas as pd
import time
import subprocess  # <--- New Import
from flask import Flask, jsonify, render_template, request
from collections import defaultdict
from pathlib import Path

from update_inventory import execute_transfer

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
        
        item['status_label'] = "STOCKOUT" if amount == 0 else "LOW STOCK"
        hospitals_in_incident.add(item['Hospital_Source'])
        
        # FIND SOLUTION
        potential_lenders = df_full_network[
            (df_full_network['ID'] == item['ID']) & 
            (df_full_network['Hospital_Source'] != item['Hospital_Source']) &
            (df_full_network['Amount'] > 0)
        ]
        
        if not potential_lenders.empty:
            best_match = potential_lenders.loc[potential_lenders['Amount'].idxmax()]
            item['solution'] = {'Facility': best_match['Hospital_Source'], 'Qty': best_match['Amount'], 'Distance': 12.5}
        else:
            item['solution'] = None
            item['ai_suggestion'] = {'M004': 'Insulin Lispro', 'M005': 'Norepinephrine', 'M009': 'Glipizide'}.get(item['ID'], "Consult Pharmacist")

        # RISK CALCULATION (1-10 Scale)
        score = float(urgency_val)
        score += 3.0 if amount == 0 else 1.0
        if not item['solution']:
            score += 2.0
            
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