import os
import json
from datetime import datetime, timezone
import pandas as pd
from geopy.distance import geodesic
import numpy as np

# --- CONFIGURAÇÕES DE AVALIAÇÃO ---
BASE_DATA_PATH = 'data/' 
YOUR_PREDICTIONS_FILE = 'resposta.json' # O arquivo gerado pelo seu main.py (na raiz do projeto)
TRUE_RESULTS_BASE_PATH = os.path.join(BASE_DATA_PATH, 'final/') # Pasta raiz para os gabaritos
EVALUATION_REPORT_FILE = 'relatorio_avaliacao_final.txt'

# --- FUNÇÕES DE CARREGAMENTO ---

def load_your_predictions(file_path):
    """Carrega as previsões do seu arquivo resposta.json."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('previsoes', [])
    except FileNotFoundError:
        print(f"ERRO: Arquivo de previsões '{file_path}' não encontrado. Certifique-se de que main.py rodou com sucesso e gerou o resposta.json.")
        return []
    except Exception as e:
        print(f"ERRO: Erro ao carregar suas previsões de '{file_path}': {e}")
        return []

def load_all_true_results(base_final_path):
    """
    Carrega todos os arquivos de resposta (gabarito) de todas as subpastas de dias em 'data/final/'.
    Assume que os arquivos de dia estão no formato 'final-YYYY-MM-DD/' e os arquivos de resposta
    dentro são 'resposta-YYYY-MM-DD_HH.json'.
    Retorna um dicionário mapeando 'id' da query para seus resultados verdadeiros.
    """
    true_results_map = {}
    print(f"Carregando resultados verdadeiros da pasta: {base_final_path}...")
    
    if not os.path.isdir(base_final_path):
        print(f"ERRO: A pasta de resultados esperados '{base_final_path}' não existe ou não é um diretório. Crie-a e adicione os arquivos de gabarito.")
        return {}

    # Itera sobre as pastas de dias (ex: 'final-2024-05-16')
    for day_folder in os.listdir(base_final_path):
        # Verifica se a pasta do dia começa com 'final-'
        if day_folder.startswith('final-') and os.path.isdir(os.path.join(base_final_path, day_folder)):
            day_path = os.path.join(base_final_path, day_folder)
            for filename in os.listdir(day_path):
                # Assume que os arquivos de gabarito são 'resposta-YYYY-MM-DD_HH.json'
                if filename.startswith('resposta-') and filename.endswith('.json'):
                    file_path = os.path.join(day_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        true_previsoes = data.get('previsoes', [])

                        for item in true_previsoes:
                            query_id = item[0]
                            if len(item) == 3: # Previsão de localização: [id, lat, lon]
                                true_results_map[query_id] = {'type': 'location', 'lat': item[1], 'lon': item[2]}
                            elif len(item) == 2: # Previsão de tempo: [id, timestamp]
                                true_results_map[query_id] = {'type': 'time', 'timestamp': item[1]}
                            # else: # Comentado para output mais limpo
                                # print(f"AVISO_GABARITO: Formato de previsão inesperado no arquivo {filename} para ID {query_id}. Pulando.")
                    except Exception as e:
                        print(f"ERRO_LOAD_TRUE_FILE: Erro ao carregar arquivo de resultado verdadeiro '{filename}': {e}")
    
    print(f"Total de {len(true_results_map)} resultados verdadeiros carregados para comparação.")
    return true_results_map

# --- FUNÇÕES DE AVALIAÇÃO ---

def calculate_errors(your_predictions, true_results_map):
    """
    Compara as previsões com os resultados verdadeiros e calcula os erros.
    """
    location_errors = [] # Erros em metros
    time_errors = []     # Erros em segundos

    for pred in your_predictions:
        pred_id = pred[0]
        true_data = true_results_map.get(pred_id)

        if not true_data:
            # print(f"AVISO: Resultado verdadeiro não encontrado para ID de previsão: {pred_id}. Pulando.")
            continue

        if len(pred) == 3 and true_data['type'] == 'location': # Previsão de localização (id, lat, lon)
            pred_lat, pred_lon = pred[1], pred[2]
            true_lat, true_lon = true_data['lat'], true_data['lon']
            
            error_distance = geodesic((pred_lat, pred_lon), (true_lat, true_lon)).meters
            location_errors.append(error_distance)

        elif len(pred) == 2 and true_data['type'] == 'time': # Previsão de tempo (id, timestamp)
            pred_timestamp = pred[1]
            true_timestamp = true_data['timestamp']
            
            error_time_ms = abs(pred_timestamp - true_timestamp)
            time_errors.append(error_time_ms / 1000) # Converte para segundos

    return location_errors, time_errors

# --- INÍCIO DO SCRIPT DE AVALIAÇÃO ---
if __name__ == "__main__":
    print("Iniciando avaliação de desempenho FINAL...")

    # 1. Carregar suas previsões (o output do seu main.py)
    your_preds = load_your_predictions(YOUR_PREDICTIONS_FILE)
    if not your_preds:
        print("Avaliação não pode ser realizada sem previsões válidas. Certifique-se de que 'resposta.json' existe e está preenchido.")
        exit()
    
    # 2. Carregar os resultados verdadeiros (gabaritos da pasta 'data/final/')
    true_results = load_all_true_results(TRUE_RESULTS_BASE_PATH)
    if not true_results:
        print("Avaliação não pode ser realizada sem resultados verdadeiros válidos. Verifique a pasta 'data/final/' e o formato dos arquivos de gabarito.")
        exit()

    # 3. Calcular os erros
    loc_errors, time_errors = calculate_errors(your_preds, true_results)

    # 4. Gerar o relatório de avaliação
    report_lines = []
    report_lines.append(f"--- Relatório de Desempenho FINAL (Comparado ao Gabarito) ---")
    report_lines.append(f"Data da Avaliação: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total de previsões geradas (seu resposta.json): {len(your_preds)}")
    report_lines.append(f"Total de resultados verdadeiros encontrados (gabarito): {len(true_results)}")
    
    report_lines.append("\n--- Métricas de Erro de Localização ---")
    if loc_errors:
        report_lines.append(f"Número de previsões de localização avaliadas: {len(loc_errors)}")
        report_lines.append(f"Erro Médio Absoluto (MAE) de Posição: {np.mean(loc_errors):.2f} metros")
        report_lines.append(f"Erro Quadrático Médio (RMSE) de Posição: {np.sqrt(np.mean(np.array(loc_errors)**2)):.2f} metros")
        report_lines.append(f"Erro Máximo de Posição: {np.max(loc_errors):.2f} metros")
    else:
        report_lines.append("Nenhuma previsão de localização para avaliar (verifique se há IDs correspondentes no gabarito).")
    
    report_lines.append("\n--- Métricas de Erro de Tempo ---")
    if time_errors:
        report_lines.append(f"Número de previsões de tempo avaliadas: {len(time_errors)}")
        report_lines.append(f"Erro Médio Absoluto (MAE) de Tempo: {np.mean(time_errors):.2f} segundos")
        report_lines.append(f"Erro Quadrático Médio (RMSE) de Tempo: {np.sqrt(np.mean(np.array(time_errors)**2)):.2f} segundos")
        report_lines.append(f"Erro Máximo de Tempo: {np.max(time_errors):.2f} segundos")
    else:
        report_lines.append("Nenhuma previsão de tempo para avaliar (verifique se há IDs correspondentes no gabarito).")

    final_report = "\n".join(report_lines)

    with open(EVALUATION_REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(final_report)
    
    print(f"\nRelatório de desempenho FINAL salvo em '{EVALUATION_REPORT_FILE}'.")
    print(final_report)
    print("\nAvaliação de desempenho FINAL concluída!")