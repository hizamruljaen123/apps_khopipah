from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster

app = Flask(__name__)

# Path ke data dan model
DATA_PATH = 'data/Data_Kecelakaan_Padang_Lawas_Utara.xlsx'
MODEL_PATH = 'models/kmeans_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
OUTPUT_EXCEL = 'data/Hasil_Klastering_Kecelakaan_Padang_Lawas_Utara.xlsx'

# Koordinat kecamatan
kecamatan_coords = {
    "Batang Onang": [1.3166932, 99.4458336],
    "Dolok": [1.6577, 98.5525],
    "Halongonan": [1.6514398, 99.6593041],
    "Padang Bolak": [3.5685536, 98.7273622],
    "Padang Bolak Julu": [1.415091, 99.5576002],
    "Portibi": [1.4375683, 99.7032047],
    "Simangambat": [1.8957231, 99.7846397],
    "Ujung Batu": [1.1896051, 99.99013],
    "Dolok Sigompulon": [1.9700433, 99.5356889],
    "Halongonan Timur": [1.6373489, 99.4558203],
    "Hulu Sihapas": [1.458091, 99.4268896],
    "Padang Bolak Tenggara": [1.54001, 99.3234082]
}

# Fungsi untuk memuat data
def load_data():
    df = pd.read_excel(DATA_PATH)
    return df

# Menghitung jarak Euclidean
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Menghitung jarak untuk setiap titik data ke centroid
def calculate_distances(df, centroid_c1, centroid_c2):
    distances_c1 = []
    distances_c2 = []
    
    for index, row in df.iterrows():
        data = np.array([row["Jumlah Kecelakaan"], row["Jumlah Meninggal"], row["Jumlah Luka Berat"], row["Jumlah Luka Ringan"]])
        distances_c1.append(euclidean_distance(data, centroid_c1))
        distances_c2.append(euclidean_distance(data, centroid_c2))
    
    return distances_c1, distances_c2

# Pembaruan Centroid
def update_centroids(df, cluster_c1, cluster_c2):
    centroid_c1 = np.mean(cluster_c1, axis=0)
    centroid_c2 = np.mean(cluster_c2, axis=0)
    return centroid_c1, centroid_c2

# Menghitung Purity
def calculate_purity(df):
    correct = 0
    total = len(df)
    
    for index, row in df.iterrows():
        if row['Cluster'] == 1:  # C1 = Rawan
            if row['Tingkat Kerawanan'] == "Rawan":
                correct += 1
        elif row['Cluster'] == 2:  # C2 = Tidak Rawan
            if row['Tingkat Kerawanan'] == "Tidak Rawan":
                correct += 1
    
    purity = correct / total
    return purity

# Fungsi untuk melatih model
def train_model():
    df = load_data()
    features = ["Jumlah Kecelakaan", "Jumlah Meninggal", "Jumlah Luka Berat", "Jumlah Luka Ringan"]
    X = df[features]
    
    # Tentukan jumlah klaster berdasarkan kebutuhan atau hasil WCSS
    k = 2  # Jumlah klaster yang sesuai dengan dokumen (Rawan dan Tidak Rawan)

    # Inisialisasi centroid manual
    centroid_c1 = np.array([169, 3, 12, 26])  # Contoh centroid C1
    centroid_c2 = np.array([55, 3, 16, 42])  # Contoh centroid C2

    # Iterasi untuk memperbarui centroid
    for i in range(10):  # Maksimal 10 iterasi
        # Menghitung jarak
        distances_c1, distances_c2 = calculate_distances(df, centroid_c1, centroid_c2)
        
        # Menetapkan klaster berdasarkan jarak
        df['Cluster'] = np.where(np.array(distances_c1) < np.array(distances_c2), 1, 2)  # C1=1, C2=2
        
        # Mengelompokkan data untuk klaster 1 dan klaster 2
        cluster_c1 = df[df['Cluster'] == 1][['Jumlah Kecelakaan', 'Jumlah Meninggal', 'Jumlah Luka Berat', 'Jumlah Luka Ringan']].values
        cluster_c2 = df[df['Cluster'] == 2][['Jumlah Kecelakaan', 'Jumlah Meninggal', 'Jumlah Luka Berat', 'Jumlah Luka Ringan']].values
        
        # Pembaruan centroid
        new_centroid_c1, new_centroid_c2 = update_centroids(df, cluster_c1, cluster_c2)
        
        # Jika centroid tidak berubah, hentikan iterasi
        if np.array_equal(new_centroid_c1, centroid_c1) and np.array_equal(new_centroid_c2, centroid_c2):
            break
        
        centroid_c1, centroid_c2 = new_centroid_c1, new_centroid_c2

    # Mapping klaster ke tingkat kerawanan
    cluster_map = {
        1: "Rawan",
        2: "Tidak Rawan"
    }
    df['Tingkat Kerawanan'] = df['Cluster'].map(cluster_map)

    # Menghitung Purity
    purity = calculate_purity(df)
    print(f"Purity: {purity * 100:.2f}%")

    # Simpan hasil ke Excel
    df.to_excel(OUTPUT_EXCEL, index=False)
    
    return df

# Rute Beranda
@app.route('/')
def index():
    return render_template('index.html')

# Rute untuk Melatih Model
@app.route('/train', methods=['GET'])
def train():
    return render_template('train.html')

@app.route('/train_model', methods=['GET'])
def train_model_route():
    df = train_model()  # Panggil fungsi train_model
    return redirect(url_for('results'))  # Redirect ke halaman results setelah model dilatih

# Rute untuk Menampilkan Hasil Klasterisasi
@app.route('/results')
def results():
    if not os.path.exists(OUTPUT_EXCEL):
        return redirect(url_for('train'))
    
    df = pd.read_excel(OUTPUT_EXCEL)
    
    # Tampilkan semua data, tidak hanya 10 baris pertama
    data = df.to_dict(orient='records')  # Convert entire DataFrame to dictionary
    return render_template('results.html', data=data)

# Rute untuk Visualisasi dengan Plotly
@app.route('/visualize')
def visualize():
    if not os.path.exists(OUTPUT_EXCEL):
        return redirect(url_for('train'))

    df = pd.read_excel(OUTPUT_EXCEL)
    
    # Visualisasi Bar Chart Tingkat Kerawanan per Tahun
    severity_by_year = df.groupby(['Tahun', 'Tingkat Kerawanan']).size().unstack(fill_value=0).reset_index()
    severity_melted = pd.melt(severity_by_year, id_vars=['Tahun'], 
                               value_vars=["Rawan", "Tidak Rawan"], 
                               var_name='Tingkat Kerawanan', value_name='Jumlah Kasus')
    
    fig_bar = px.bar(severity_melted, x='Tahun', y='Jumlah Kasus', color='Tingkat Kerawanan',
                     title='Jumlah Tingkat Keparahan Kecelakaan Per Tahun',
                     labels={'Jumlah Kasus': 'Jumlah Kasus'},
                     height=600, 
                     width=1000, 
                     text='Jumlah Kasus',
                     category_orders={'Tingkat Kerawanan': ["Rawan", "Tidak Rawan"]})
    
    fig_bar.update_traces(texttemplate='%{text}', textposition='outside', 
                          marker=dict(line=dict(color='rgb(0,0,0)', width=1.5)))
    
    fig_bar.update_layout(xaxis_title='Tahun',
                          yaxis_title='Jumlah Kasus',
                          legend_title='Tingkat Kerawanan',
                          barmode='stack')
    
    graph_html = fig_bar.to_html(full_html=False)
    
    return render_template('visualize.html', graph_html=graph_html)

# Rute untuk Menampilkan Peta Folium dengan Pemilihan Tahun
@app.route('/map', methods=['GET', 'POST'])
def map_view():
    if not os.path.exists(OUTPUT_EXCEL):
        return redirect(url_for('train'))
    
    df = pd.read_excel(OUTPUT_EXCEL)
    
    # Mendapatkan daftar tahun yang tersedia
    available_years = sorted(df['Tahun'].unique())
    
    if request.method == 'POST':
        selected_year = int(request.form.get('year'))
        return redirect(url_for('map_view', year=selected_year))
    else:
        selected_year = request.args.get('year', default=None, type=int)
    
    if selected_year:
        df_year = df[df['Tahun'] == selected_year]
        title = f"Tingkat Kerawanan Kecelakaan Tahun {selected_year}"
    else:
        selected_year = df['Tahun'].max()
        df_year = df[df['Tahun'] == selected_year]
        title = f"Tingkat Kerawanan Kecelakaan Tahun {selected_year}"
    
    # Tambahkan koordinat
    df_year['Coordinates'] = df_year['Kecamatan'].map(kecamatan_coords)
    
    df_year = df_year[df_year['Coordinates'].notna()]
    
    color_map = {
        "Rawan": "orange",
        "Tidak Rawan": "green"
    }
    
    m = folium.Map(location=[1.4099, 99.7092], zoom_start=10)
    marker_cluster = MarkerCluster().add_to(m)
    
    for _, row in df_year.iterrows():
        coords = row['Coordinates']
        folium.Marker(
            location=coords,
            popup=f"{row['Kecamatan']}: {row['Jumlah Kecelakaan']} kasus, Tingkat Kerawanan: {row['Tingkat Kerawanan']}",
            icon=folium.Icon(color=color_map.get(row['Tingkat Kerawanan'], 'blue'))
        ).add_to(marker_cluster)
    
    map_html = m._repr_html_()
    
    return render_template('map.html', map_html=map_html, available_years=available_years, selected_year=selected_year, title=title)

# Rute untuk Mengunduh Hasil Klasterisasi
@app.route('/download')
def download():
    if not os.path.exists(OUTPUT_EXCEL):
        return redirect(url_for('train'))
    return send_file(OUTPUT_EXCEL, as_attachment=True)

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    app.run(debug=True)
