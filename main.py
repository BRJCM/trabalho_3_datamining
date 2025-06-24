import os
import json
import pandas as pd
from datetime import datetime, timedelta, timezone 
from geopy.distance import geodesic 

# --- CONFIGURAÇÕES GLOBAIS ---
ALUNO_NOME = "Brian Medeiros"
SENHA_API = "abc123"

# Caminhos das pastas (AJUSTADOS PARA NOVA ESTRUTURA)
BASE_DATA_PATH = 'data/' 
OUTPUT_FILE = 'resposta.json'
EVAL_REPORT_FILE = 'relatorio_avaliacao_mvp.txt' 

# Linhas de ônibus a serem consideradas
LINHAS_INTERESSE = [
    "483", "864", "639", "3", "309", "774", "629", "371", "397", "100", "838",
    "315", "624", "388", "918", "665", "328", "497", "878", "355", "138",
    "606", "457", "550", "803", "917", "638", "2336", "399", "298", "867",
    "553", "565", "422", "756", "186012003", "292", "554", "634", "232",
    "415", "2803", "324", "852", "557", "759", "343", "779", "905", "108"
]

# Cache para mapear (ano, mês, dia, hora) para lista de caminhos de arquivo histórico bruto
HISTORICAL_RAW_FILE_PATH_CACHE = {}

# Cache de DataFrames carregados para a janela de um dia de teste específico
CURRENT_TEST_DAY_DATA_CACHE = {} 

# --- FUNÇÕES UTILITÁRIAS ---

def load_and_preprocess_single_raw_file(file_path):
    """
    Carrega um único arquivo JSON de dados brutos, limpa e padroniza os dados.
    Aplica filtro inicial por linhas de interesse e horário para reduzir volume.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)

        initial_len = len(df) 
        
        df = df[df['linha'].isin(LINHAS_INTERESSE)].copy() 
        if len(df) < initial_len:
            # print(f"  INFO_LOAD_FILTER: Descartados {initial_len - len(df)} registros de linhas não-interesse em {os.path.basename(file_path)}.")
            pass 

        if df.empty:
            return pd.DataFrame()

        required_cols = ['ordem', 'latitude', 'longitude', 'datahoraservidor', 'linha', 'datahora', 'velocidade']
        for col in required_cols:
            if col not in df.columns:
                df[col] = pd.NA

        df['latitude'] = df['latitude'].astype(str).str.replace(',', '.', regex=False).astype(float)
        df['longitude'] = df['longitude'].astype(str).str.replace(',', '.', regex=False).astype(float)
        df['velocidade'] = pd.to_numeric(df['velocidade'], errors='coerce').fillna(0)
        
        df['timestamp_ms'] = pd.to_numeric(df['datahoraservidor'], errors='coerce')
        df['timestamp_ms'] = df['timestamp_ms'].fillna(pd.to_numeric(df['datahora'], errors='coerce'))
        
        df = df.dropna(subset=['timestamp_ms', 'latitude', 'longitude'])
        
        df['datahoraservidor_dt'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)

        len_after_line_filter = len(df)
        df = df[(df['datahoraservidor_dt'].dt.hour >= 8) & (df['datahoraservidor_dt'].dt.hour < 23)]
        if len(df) < len_after_line_filter:
            pass 
        
        if not df.empty:
            print(f"  INFO_LOADED_FILE: Arquivo {os.path.basename(file_path)} carregado e pré-filtrado: {len(df)} registros válidos.")

        return df

    except Exception as e:
        print(f"ERRO_LOAD_RAW_FILE: Erro ao processar o arquivo {file_path}: {e}") 
        return pd.DataFrame()

def load_test_queries_file(file_path):
    """Carrega as queries do arquivo de teste (treino-YYYY-MM-DD_HH.json)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            if 'datahora' in item:
                item['datahora_dt'] = datetime.fromtimestamp(int(item['datahora']) / 1000, tz=timezone.utc)
            if 'latitude' in item: 
                item['latitude'] = float(str(item['latitude']).replace(',', '.'))
            if 'longitude' in item: 
                item['longitude'] = float(str(item['longitude']).replace(',', '.'))
        return data
    except Exception as e:
        print(f"ERRO_LOAD_QUERY: Erro ao carregar queries de teste de {file_path}: {e}")
        return []

def build_historical_file_path_cache(base_historical_path):
    """
    Escaneia a pasta de dados históricos (com subpastas de dias) UMA ÚNICA VEZ
    e cria um cache (ano, mês, dia, hora) -> lista de caminhos de arquivo.
    """
    print(f"Construindo cache de caminhos de arquivos históricos em: {base_historical_path}...")
    file_count = 0
    latest_historical_date = None 
    if not os.path.isdir(base_historical_path):
        print(f"ERRO: A pasta histórica '{base_historical_path}' não existe ou não é um diretório. Verifique o caminho.")
        return

    for day_folder in os.listdir(base_historical_path):
        day_path = os.path.join(base_historical_path, day_folder)
        if os.path.isdir(day_path):
            try: 
                current_day_dt = datetime.strptime(day_folder, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if latest_historical_date is None or current_day_dt > latest_historical_date:
                    latest_historical_date = current_day_dt
            except ValueError:
                continue

            for filename in os.listdir(day_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(day_path, filename)
                    try:
                        name_parts = os.path.splitext(filename)[0].split('_')
                        if len(name_parts) == 2:
                            date_part = name_parts[0]
                            hour_part = name_parts[1]
                            
                            file_dt = datetime.strptime(f"{date_part}_{hour_part}", "%Y-%m-%d_%H").replace(tzinfo=timezone.utc)
                            key = (file_dt.year, file_dt.month, file_dt.day, file_dt.hour)
                            
                            if key not in HISTORICAL_RAW_FILE_PATH_CACHE:
                                HISTORICAL_RAW_FILE_PATH_CACHE[key] = []
                            HISTORICAL_RAW_FILE_PATH_CACHE[key].append(file_path)
                            file_count += 1
                    except ValueError:
                        pass 
                    except Exception as e:
                        pass 
    print(f"Cache de caminhos de arquivos históricos construído. {len(HISTORICAL_RAW_FILE_PATH_CACHE)} horas de dados mapeadas de {file_count} arquivos.")
    return latest_historical_date 


# Cache de DataFrames carregados para a janela de um dia de teste específico
CURRENT_TEST_DAY_DATA_CACHE = {} 

def load_historical_data_for_test_day_window(test_day_datetime_ref, hours_before=12): # Aumentado para 12 horas para garantir
    """
    Carrega TODOS os DataFrames de dados brutos (normais) relevantes para a janela de um DIA de teste.
    Popula o CURRENT_TEST_DAY_DATA_CACHE para evitar recarregar arquivos grandes repetidamente.
    A janela de carregamento é maior para garantir que haja dados para todas as queries do dia de teste.
    """
    global CURRENT_TEST_DAY_DATA_CACHE
    CURRENT_TEST_DAY_DATA_CACHE = {} 

    # A janela de tempo para carregamento abrange o dia inteiro do teste e algumas horas antes
    start_window_for_load = test_day_datetime_ref.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(hours=hours_before)
    
    # end_window_for_load vai até o final daquele dia (23:59:59)
    end_window_for_load = test_day_datetime_ref.replace(hour=23, minute=59, second=59, microsecond=999999) 

    print(f"  INFO_DAY_LOAD: Pré-carregando dados brutos para a janela do dia de teste: {start_window_for_load.strftime('%Y-%m-%d %H:%M:%S UTC')} a {end_window_for_load.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    files_to_load_paths_for_window = []
    current_dt_iterator = start_window_for_load.replace(minute=0, second=0, microsecond=0)
    
    while current_dt_iterator <= end_window_for_load.replace(minute=0, second=0, microsecond=0): 
        key = (current_dt_iterator.year, current_dt_iterator.month, current_dt_iterator.day, current_dt_iterator.hour)
        if key in HISTORICAL_RAW_FILE_PATH_CACHE:
            files_to_load_paths_for_window.extend(HISTORICAL_RAW_FILE_PATH_CACHE[key])
        current_dt_iterator += timedelta(hours=1) 
    
    unique_files_to_load = sorted(list(set(files_to_load_paths_for_window)))
    print(f"  INFO_DAY_LOAD: Total de {len(unique_files_to_load)} arquivos brutos identificados para pré-carregamento.")

    for file_path in unique_files_to_load:
        df = load_and_preprocess_single_raw_file(file_path)
        if not df.empty:
            file_dt_key_from_path = datetime.strptime(os.path.basename(file_path).replace('.json',''), "%Y-%m-%d_%H").replace(tzinfo=timezone.utc)
            key = (file_dt_key_from_path.year, file_dt_key_from_path.month, file_dt_key_from_path.day, file_dt_key_from_path.hour)
            CURRENT_TEST_DAY_DATA_CACHE[key] = df
    print(f"  INFO_DAY_LOAD: Pré-carregamento de dados brutos do dia concluído. {len(CURRENT_TEST_DAY_DATA_CACHE)} horas de DataFrames carregados para o cache do dia de teste.")


def get_recent_historical_data_for_query_from_cache(ordem, linha, query_datetime, hours_before=5): 
    """
    Recupera dados históricos relevantes do CURRENT_TEST_DAY_DATA_CACHE (já pré-carregados para o dia do teste).
    Aplica filtros de ônibus/linha e janela de tempo específica da query.
    """
    recent_data_dfs = []
    
    end_window_query = query_datetime
    start_window_query = query_datetime - timedelta(hours=hours_before)

    current_dt_iterator = start_window_query.replace(minute=0, second=0, microsecond=0) 
    
    while current_dt_iterator <= end_window_query.replace(minute=0, second=0, microsecond=0): 
        key = (current_dt_iterator.year, current_dt_iterator.month, current_dt_iterator.day, current_dt_iterator.hour)
        if key in CURRENT_TEST_DAY_DATA_CACHE:
            recent_data_dfs.append(CURRENT_TEST_DAY_DATA_CACHE[key])
        current_dt_iterator += timedelta(hours=1) 
    
    if not recent_data_dfs:
        # print(f"    AVISO_QUERY: Nenhum DataFrame para horas {start_window_query.hour}-{end_window_query.hour} no cache do dia para {ordem}/{linha}.")
        return []

    full_recent_df = pd.concat(recent_data_dfs, ignore_index=True)
    
    if not all(col in full_recent_df.columns for col in ['ordem', 'linha', 'datahoraservidor_dt']):
        # print(f"    AVISO_QUERY: Colunas essenciais ausentes no full_recent_df para {ordem}/{linha}.")
        return []

    filtered_df = full_recent_df[
        (full_recent_df['datahoraservidor_dt'] >= start_window_query) & 
        (full_recent_df['datahoraservidor_dt'] <= end_window_query) & 
        (full_recent_df['ordem'] == ordem) & 
        (full_recent_df['linha'] == linha) &
        (full_recent_df['linha'].isin(LINHAS_INTERESSE)) 
    ].sort_values(by='datahoraservidor_dt').reset_index(drop=True)

    if filtered_df.empty:
        # print(f"    AVISO_QUERY: DataFrame vazio após filtros finais para {ordem}/{linha} na janela {start_window_query.strftime('%H:%M')}-{end_window_query.strftime('%H:%M')}. Dados ausentes ou filtrados.")
        return []

    # print(f"    INFO_QUERY_SUCCESS: Histórico encontrado para {ordem}/{linha}: {len(filtered_df)} registros.")
    return filtered_df[['latitude', 'longitude', 'timestamp_ms', 'datahoraservidor_dt', 'velocidade']].to_dict('records')

# --- FUNÇÕES DE PREVISÃO (MVP SIMPLIFICADO) ---

def predict_location(bus_history, target_timestamp_ms):
    """
    Prevê a localização (lat, lon) de um ônibus dado um timestamp futuro.
    Usa interpolação linear simples ou a última posição conhecida.
    """
    if not bus_history:
        return None, None 

    target_datetime = datetime.fromtimestamp(target_timestamp_ms / 1000, tz=timezone.utc)

    prev_point = None
    next_point = None
    
    for i in range(len(bus_history)):
        current_point = bus_history[i]
        current_datetime = current_point['datahoraservidor_dt']

        if current_datetime <= target_datetime:
            prev_point = current_point
        else:
            next_point = current_point
            break 

    if prev_point is None and next_point is not None:
        return next_point['latitude'], next_point['longitude']
    
    if prev_point is not None and next_point is not None:
        time_diff = (next_point['timestamp_ms'] - prev_point['timestamp_ms'])
        if time_diff == 0: 
            return prev_point['latitude'], prev_point['longitude']

        time_ratio = (target_timestamp_ms - prev_point['timestamp_ms']) / time_diff

        lat = prev_point['latitude'] + time_ratio * (next_point['latitude'] - prev_point['latitude'])
        lon = prev_point['longitude'] + time_ratio * (next_point['longitude'] - prev_point['longitude'])
        return lat, lon

    if prev_point is not None and next_point is None:
        return prev_point['latitude'], prev_point['longitude']

    if len(bus_history) == 1 and prev_point is not None:
        return prev_point['latitude'], prev_point['longitude']

    return None, None 


def predict_arrival_time(bus_history, target_location):
    """
    Prevê o timestamp de chegada em uma localização (lat, lon) alvo.
    Para MVP, encontra o ponto histórico mais próximo no trajeto.
    """
    if not bus_history or not target_location:
        return None

    target_lat, target_lon = target_location['latitude'], target_location['longitude']
    
    min_distance = float('inf')
    closest_point_timestamp = None
    
    for point in bus_history:
        point_lat, point_lon = point['latitude'], point['longitude']
        dist = geodesic((point_lat, point_lon), (target_lat, target_lon)).meters
        
        if dist < min_distance:
            min_distance = dist
            closest_point_timestamp = point['timestamp_ms']
            
    return closest_point_timestamp


# --- FUNÇÕES DE AVALIAÇÃO (MVP SIMPLIFICADO) ---
def evaluate_predictions_mvp(previsoes, all_test_queries_list_for_eval):
    """
    Avalia as previsões comparando-as com o histórico conhecido.
    Retorna uma string de relatório.
    """
    total_pos_error = 0.0 
    total_time_error_sec = 0.0 
    pos_predictions_count = 0
    time_predictions_count = 0
    
    evaluation_report = ["--- Relatório de Avaliação MVP (Estimado) ---"]
    evaluation_report.append("Nota: Esta é uma avaliação interna simplificada, pois não temos um arquivo 'resposta' real para comparação direta aqui.")
    evaluation_report.append("Os erros são calculados comparando a previsão com o ponto histórico mais próximo ou estimado na janela de dados.")
    
    queries_by_id = {q['id']: q for q in all_test_queries_list_for_eval}

    for pred in previsoes:
        query_id = pred[0]
        original_query = queries_by_id.get(query_id)

        if not original_query:
            continue

        ordem_bus = original_query['ordem']
        linha_bus = original_query['linha']
        query_datetime = original_query.get('datahora_dt')

        if query_datetime is None and 'id_arquivo_teste' in original_query:
             try:
                base_test_filename = os.path.basename(original_query['id_arquivo_teste'])
                parts = os.path.splitext(base_test_filename)[0].split('_') 
                date_part = parts[0].replace('treino-', '') 
                hour_part = parts[1]
                query_datetime = datetime.strptime(f"{date_part}_{hour_part}", "%Y-%m-%d_%H").replace(tzinfo=timezone.utc)
             except Exception:
                 query_datetime = datetime.now(timezone.utc) 
        elif query_datetime is None: 
             query_datetime = datetime.now(timezone.utc)

        
        eval_hours_before = 5 
        eval_end_window = query_datetime
        eval_start_window = query_datetime - timedelta(hours=eval_hours_before)
        eval_start_window = eval_start_window.replace(minute=0, second=0, microsecond=0)

        eval_dfs_for_query = []
        eval_current_dt_iterator = eval_start_window
        while eval_current_dt_iterator <= eval_end_window.replace(minute=0, second=0, microsecond=0):
            key = (eval_current_dt_iterator.year, eval_current_dt_iterator.month, eval_current_dt_iterator.day, eval_current_dt_iterator.hour)
            if key in HISTORICAL_RAW_FILE_PATH_CACHE:
                for path in HISTORICAL_RAW_FILE_PATH_CACHE[key]:
                    df = load_and_preprocess_single_raw_file(path) 
                    if not df.empty:
                        eval_dfs_for_query.append(df)
            eval_current_dt_iterator += timedelta(hours=1)

        temp_full_eval_df = pd.concat(eval_dfs_for_query, ignore_index=True) if eval_dfs_for_query else pd.DataFrame()

        bus_history_for_eval = temp_full_eval_df[
            (temp_full_eval_df['datahoraservidor_dt'] >= eval_start_window) & 
            (temp_full_eval_df['datahoraservidor_dt'] <= eval_end_window) & 
            (temp_full_eval_df['ordem'] == ordem) & 
            (temp_full_eval_df['linha'] == linha) &
            (temp_full_eval_df['linha'].isin(LINHAS_INTERESSE)) 
        ].sort_values(by='datahoraservidor_dt').to_dict('records')

        if not bus_history_for_eval:
            continue

        if len(pred) == 3: # Previsão de localização (id, lat, lon)
            pred_lat, pred_lon = pred[1], pred[2]
            
            target_timestamp = original_query.get('datahora') 
            if target_timestamp:
                closest_historical_point = None
                min_time_diff = float('inf')
                for point in bus_history_for_eval:
                    time_diff = abs(point['timestamp_ms'] - target_timestamp)
                    if time_diff < min_time_diff: 
                        min_time_diff = time_diff
                        closest_historical_point = point
                
                if closest_historical_point:
                    true_lat, true_lon = closest_historical_point['latitude'], closest_historical_point['longitude']
                    error_dist = geodesic((pred_lat, pred_lon), (true_lat, true_lon)).meters
                    total_pos_error += error_dist
                    pos_predictions_count += 1

        elif len(pred) == 2: # Previsão de tempo (id, timestamp)
            pred_timestamp = pred[1]
            target_location = {'latitude': original_query['latitude'], 'longitude': original_query['longitude']}

            closest_historical_point = None
            min_distance = float('inf')
            
            for point in bus_history_for_eval:
                point_lat, point_lon = point['latitude'], point['longitude']
                dist = geodesic((point_lat, point_lon), (target_location['latitude'], target_location['longitude'])).meters
                if dist < min_distance:
                    min_distance = dist
                    closest_historical_point = point
            
            if closest_historical_point: 
                true_timestamp = closest_historical_point['timestamp_ms']
                error_time_ms = abs(pred_timestamp - true_timestamp)
                total_time_error_sec += (error_time_ms / 1000) 
                time_predictions_count += 1

    if pos_predictions_count > 0:
        avg_pos_error = total_pos_error / pos_predictions_count
        evaluation_report.append(f"\nErro médio de posição (MAE estimado): {avg_pos_error:.2f} metros em {pos_predictions_count} previsões.")
    else:
        evaluation_report.append("\nNenhuma previsão de posição para avaliar.")

    if time_predictions_count > 0:
        avg_time_error_sec_total = total_time_error_sec / time_predictions_count
        evaluation_report.append(f"Erro médio de tempo (MAE estimado): {avg_time_error_sec_total:.2f} segundos em {time_predictions_count} previsões.")
    else:
        evaluation_report.append("Nenhuma previsão de tempo para avaliar.")
    
    return "\n".join(evaluation_report)


# --- INÍCIO DO SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    print("Iniciando o processamento principal...")

    # PASSO 1: Construir o cache de caminhos de arquivos históricos (de toda a pasta historical/)
    latest_hist_date = build_historical_file_path_cache(os.path.join(BASE_DATA_PATH, 'historical'))

    previsoes_finais = []
    all_test_queries_for_eval = [] 
    
    test_days_folders = sorted([d for d in os.listdir(os.path.join(BASE_DATA_PATH, 'test')) if os.path.isdir(os.path.join(BASE_DATA_PATH, 'test', d))])

    # AVISO: VERIFICAÇÃO CRÍTICA DE DADOS
    if test_days_folders and latest_hist_date:
        earliest_test_date = datetime.strptime(test_days_folders[0], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        # Se o início dos dados de teste é após o fim dos dados históricos
        if earliest_test_date > latest_hist_date + timedelta(days=1): # Adiciona 1 dia de margem
            print(f"\nERRO CRÍTICO DE DADOS: A pasta 'historical/' tem dados apenas até {latest_hist_date.strftime('%Y-%m-%d')}.")
            print(f"Os testes começam em {earliest_test_date.strftime('%Y-%m-%d')}.")
            print("Não há sobreposição de datas. O modelo não encontrará histórico para as previsões.")
            print("Por favor, adicione arquivos de dados históricos na pasta 'data/historical/' para cobrir as datas dos seus testes.")
            exit() 


    for day_folder in test_days_folders:
        current_test_day_path = os.path.join(BASE_DATA_PATH, 'test', day_folder)
        
        test_day_base_datetime = None
        try:
            test_day_base_datetime = datetime.strptime(day_folder, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception as e:
            print(f"  ERRO_MAIN: Não foi possível inferir data base da pasta de teste {day_folder}. Pulando. Erro: {e}")
            continue

        print(f"\nProcessando dados para o dia de teste: {day_folder}")

        # PASSO 2: Pré-carregar DataFrames brutos relevantes para ESTE DIA de teste no cache de dia
        # Esta função populará o CURRENT_TEST_DAY_DATA_CACHE para o arquivo de teste atual
        # Aumentei a janela de busca para a carga inicial para 5 horas
        load_historical_data_for_test_day_window(test_day_base_datetime, hours_before=5) 

        test_query_files = sorted([f for f in os.listdir(current_test_day_path) if f.startswith('treino-') and f.endswith('.json')])

        for test_filename in test_query_files:
            test_file_path = os.path.join(current_test_day_path, test_filename)
            print(f"  Processando arquivo de query: {os.path.basename(test_file_path)}")
            
            test_queries = load_test_queries_file(test_file_path)
            
            for query in test_queries:
                query['id_arquivo_teste'] = test_file_path 
                all_test_queries_for_eval.append(query) 

            for query in test_queries:
                query_id = query['id']
                ordem_bus = query['ordem']
                linha_bus = query['linha']
                query_datetime = query.get('datahora_dt') 

                if query_datetime is None:
                    try:
                        parts = os.path.splitext(os.path.basename(test_filename))[0].split('_') 
                        date_part = parts[0].replace('treino-', '') 
                        hour_part = parts[1]
                        query_datetime = datetime.strptime(f"{date_part}_{hour_part}", "%Y-%m-%d_%H").replace(tzinfo=timezone.utc)
                    except Exception as e:
                        query_datetime = datetime.now(timezone.utc)

                bus_history = get_recent_historical_data_for_query_from_cache(
                    ordem_bus, linha_bus, query_datetime, hours_before=5
                ) 
                
                if not bus_history:
                    continue 
                
                if 'datahora' in query and 'latitude' not in query and 'longitude' not in query:
                    target_timestamp_ms = query['datahora']
                    pred_lat, pred_lon = predict_location(bus_history, target_timestamp_ms)
                    
                    if pred_lat is not None and pred_lon is not None:
                        previsoes_finais.append([query_id, round(pred_lat, 5), round(pred_lon, 5)])

                elif 'latitude' in query and 'longitude' in query and 'datahora' not in query:
                    target_location = {'latitude': query['latitude'], 'longitude': query['longitude']}
                    pred_timestamp = predict_arrival_time(bus_history, target_location)
                    
                    if pred_timestamp is not None:
                        previsoes_finais.append([query_id, pred_timestamp])

    # --- GERAÇÃO DO ARQUIVO resposta.json ---
    final_response = {
        "aluno": ALUNO_NOME,
        "datahora": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), 
        "previsoes": previsoes_finais,
        "senha": SENHA_API
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_response, f, ensure_ascii=False, indent=2)

    print(f"\nArquivo '{OUTPUT_FILE}' gerado com sucesso! Total de previsões: {len(previsoes_finais)}")
    print("Previsões geradas (primeiras 5):")
    for p in previsoes_finais[:5]:
        print(f"  {p}")
    if len(previsoes_finais) > 5:
        print(f"  ...e mais {len(previsoes_finais) - 5} previsões.")

    # --- AVALIAÇÃO DO MVP (usando a função do evaluate.py) ---
    print("\nRealizando avaliação interna do MVP...")
    evaluation_report_str = evaluate_predictions_mvp(previsoes_finais, all_test_queries_for_eval)

    with open(EVAL_REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(evaluation_report_str)
    print(f"Relatório de avaliação salvo em '{EVAL_REPORT_FILE}'.")
    print(evaluation_report_str)


    # --- INSTRUÇÕES PARA ENVIO À API ---
    print("\n--- PRÓXIMO PASSO: ENVIAR O ARQUIVO PARA A API ---")
    print("Abra o terminal na pasta 'seu_trabalho_onibus/' e execute o seguinte comando:")
    print(f"curl -X 'POST' \\")
    print(f"  'https://barra.cos.ufrj.br:443/datamining/rpc/avalia' \\")
    print(f"  -H 'accept: application/json' \\")
    print(f"  -H 'Content-Type: application/json' \\")
    print(f"  -d @{OUTPUT_FILE}")
    print("\nCertifique-se de que o 'resposta.json' esteja na mesma pasta onde você executa o 'curl'.")
    print("Boa sorte com o trabalho!")