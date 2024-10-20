from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster
import io

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

# Fungsi untuk melatih model
def train_model():
    df = load_data()
    features = ["Jumlah Kecelakaan", "Jumlah Meninggal", "Jumlah Luka Berat", "Jumlah Luka Ringan"]
    X = df[features]
    
    # Normalisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Tentukan jumlah klaster
    k = 4  # Aman, Berpotensi Rawan, Rawan, Sangat Rawan
    
    # Terapkan K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    df['Cluster'] = kmeans.labels_
    
    # Simpan scaler dan model
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(kmeans, MODEL_PATH)
    
    # Mapping klaster ke tingkat kerawanan
    cluster_map = {
        0: "Aman",
        1: "Berpotensi Rawan",
        2: "Rawan",
        3: "Sangat Rawan"
    }
    df['Tingkat Kerawanan'] = df['Cluster'].map(cluster_map)
    
    # Simpan hasil ke Excel
    df.to_excel(OUTPUT_EXCEL, index=False)
    
    return df

# Rute Beranda
@app.route('/')
def index():
    return render_template('index.html')

# Rute untuk Melatih Model
@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        df = train_model()
        return redirect(url_for('results'))
    return render_template('train.html')

# Rute untuk Menampilkan Hasil Klasterisasi
@app.route('/results')
def results():
    if not os.path.exists(OUTPUT_EXCEL):
        return redirect(url_for('train'))
    df = pd.read_excel(OUTPUT_EXCEL)
    # Tampilkan 10 baris pertama
    data = df.head(10).to_dict(orient='records')
    return render_template('results.html', tables=data, titles=['Hasil Klasterisasi'])

# Rute untuk Visualisasi dengan Plotly
@app.route('/visualize')
def visualize():
    if not os.path.exists(OUTPUT_EXCEL):
        return redirect(url_for('train'))
    df = pd.read_excel(OUTPUT_EXCEL)
    
    # **Bagian 1: Visualisasi Bar Chart Tingkat Kerawanan per Tahun**
    
    # Group by year and severity level
    severity_by_year = df.groupby(['Tahun', 'Tingkat Kerawanan']).size().unstack(fill_value=0).reset_index()
    
    # Melt the DataFrame untuk Plotly
    severity_melted = pd.melt(severity_by_year, id_vars=['Tahun'], value_vars=["Aman", "Berpotensi Rawan", "Rawan", "Sangat Rawan"], 
                              var_name='Tingkat Kerawanan', value_name='Jumlah Kasus')
    
    # Visualisasi dengan Plotly Bar Chart
    fig_bar = px.bar(severity_melted, x='Tahun', y='Jumlah Kasus', color='Tingkat Kerawanan',
                     title='Jumlah Tingkat Keparahan Kecelakaan Per Tahun',
                     labels={'Jumlah Kasus': 'Jumlah Kasus'},
                     height=600, 
                     width=1000, 
                     text='Jumlah Kasus',
                     category_orders={'Tingkat Kerawanan': ["Aman", "Berpotensi Rawan", "Rawan", "Sangat Rawan"]})
    
    # Tambahkan label pada bar chart
    fig_bar.update_traces(texttemplate='%{text}', textposition='outside', 
                          marker=dict(line=dict(color='rgb(0,0,0)', width=1.5)))
    
    # Update layout
    fig_bar.update_layout(xaxis_title='Tahun',
                          yaxis_title='Jumlah Kasus',
                          legend_title='Tingkat Kerawanan',
                          barmode='stack')
    
    graph_html = fig_bar.to_html(full_html=False)
    
    # **Bagian 2: Visualisasi Tabel Tingkat Kerawanan per Kecamatan dan Tahun**
    
    # Pivot tabel untuk Plotly
    table_pivot = df.pivot_table(index='Kecamatan', columns='Tahun', values='Tingkat Kerawanan', aggfunc=lambda x: ', '.join(x.unique()))
    table_pivot.reset_index(inplace=True)
    
    # Pastikan urutan kolom tahun
    years = sorted(df['Tahun'].unique())
    columns = ['Kecamatan'] + years
    
    # Menyesuaikan urutan kolom
    table_pivot = table_pivot.reindex(columns=columns)
    
    # Membuat data untuk Plotly Table
    header = dict(values=table_pivot.columns.tolist(),
                  fill_color='paleturquoise',
                  align='left')
    
    cells = dict(values=[table_pivot[col].tolist() for col in table_pivot.columns],
                 fill_color='lavender',
                 align='left')
    
    fig_table = go.Figure(data=[go.Table(header=header, cells=cells)])
    
    fig_table.update_layout(title='Tingkat Kerawanan per Kecamatan dan Tahun')
    
    table_html = fig_table.to_html(full_html=False)
    
    return render_template('visualize.html', graph_html=graph_html, table_html=table_html)

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
        # Jika tidak ada tahun yang dipilih, tampilkan peta tanpa filter atau dengan tahun terbaru
        selected_year = df['Tahun'].max()
        df_year = df[df['Tahun'] == selected_year]
        title = f"Tingkat Kerawanan Kecelakaan Tahun {selected_year}"
    
    # Tambahkan koordinat
    df_year['Coordinates'] = df_year['Kecamatan'].map(kecamatan_coords)
    
    # Hanya baris dengan koordinat valid
    df_year = df_year[df_year['Coordinates'].notna()]
    
    # Warna untuk setiap tingkat kerawanan
    color_map = {
        "Aman": "green",
        "Berpotensi Rawan": "yellow",
        "Rawan": "orange",
        "Sangat Rawan": "red"
    }
    
    # Buat peta dasar
    m = folium.Map(location=[1.4099, 99.7092], zoom_start=10)
    marker_cluster = MarkerCluster().add_to(m)
    
    # Tambahkan marker
    for _, row in df_year.iterrows():
        coords = row['Coordinates']
        folium.Marker(
            location=coords,
            popup=f"{row['Kecamatan']}: {row['Jumlah Kecelakaan']} kasus, Tingkat Kerawanan: {row['Tingkat Kerawanan']}",
            icon=folium.Icon(color=color_map.get(row['Tingkat Kerawanan'], 'blue'))
        ).add_to(marker_cluster)
    
    # Render Folium map sebagai HTML
    map_html = m._repr_html_()
    
    return render_template('map.html', map_html=map_html, available_years=available_years, selected_year=selected_year, title=title)

# Rute untuk Mengunduh Hasil Klasterisasi
@app.route('/download')
def download():
    if not os.path.exists(OUTPUT_EXCEL):
        return redirect(url_for('train'))
    return send_file(OUTPUT_EXCEL, as_attachment=True)

if __name__ == '__main__':
    # Pastikan direktori models dan data ada
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    app.run(debug=True)
