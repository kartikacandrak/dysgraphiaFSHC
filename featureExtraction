import os

# List isi folder
folder_path = '/content/drive/MyDrive/DysgraphiaDB1'
excel_path ='/content/drive/MyDrive/DysgraphiaDB1/data2_SciRep_pub.xlsx'
svc_files = [f for f in os.listdir(folder_path) if f.endswith('.svc')]
#print("Ditemukan file:", svc_files)

# === 2️⃣ Baca file Excel untuk metadata diagnosa ===
df_info = pd.read_excel(excel_path)

# Pastikan ada kolom 'filename'. Jika tidak ada, buat dari ID
if 'filename' not in df_info.columns and 'ID' in df_info.columns:
    df_info['filename'] = df_info['ID'].apply(lambda x: f"u{int(x):05d}s0001_hw0001.svc")

# Ambil hanya kolom penting
df_info = df_info[['filename', 'diag', 'age', 'sex', 'hand']]

#FEATURE EXTRACTION

col_names = ['x', 'y', 'time', 'penStatus', 'azimuth', 'altitude', 'pressure']

summary_stats = []

svc_files = [f for f in os.listdir(folder_path) if f.endswith('.svc')]

for fn in svc_files:
    file_path = os.path.join(folder_path, fn)

    try:

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.read().splitlines()

        data_lines = lines[1:]
        rows = [line.strip().split() for line in data_lines if line.strip()]
        df = pd.DataFrame(rows, columns=col_names)
        df = df.apply(pd.to_numeric, errors='coerce')

        # Delta posisi & waktu
        df['dx'] = df['x'].diff()
        df['dy'] = df['y'].diff()
        df['dt'] = df['time'].diff()
        df = df[df['dt'] > 0]

        # === VELOCITY ===
        df['v_x'] = df['dx'] / df['dt']
        df['v_y'] = df['dy'] / df['dt']
        df['v_total'] = np.sqrt(df['v_x']**2 + df['v_y']**2)

        # === ACCELERATION ===
        df['dvx'] = df['v_x'].diff()
        df['dvy'] = df['v_y'].diff()
        df['dt_a'] = df['dt'].shift(-1)
        df = df[df['dt_a'] > 0]
        df['a_x'] = df['dvx'] / df['dt_a']
        df['a_y'] = df['dvy'] / df['dt_a']
        df['a_total'] = np.sqrt(df['a_x']**2 + df['a_y']**2)

        # === JERK ===
        df['dax'] = df['a_x'].diff()
        df['day'] = df['a_y'].diff()
        df['dt_j'] = df['dt_a'].shift(-1)
        df = df[df['dt_j'] > 0]
        df['j_x'] = df['dax'] / df['dt_j']
        df['j_y'] = df['day'] / df['dt_j']
        df['j_total'] = np.sqrt(df['j_x']**2 + df['j_y']**2)

        # === LENGTH OF SEGMENT ===
        df['segment_len_x'] = df['dx'].abs()
        df['segment_len_y'] = df['dy'].abs()
        df['segment_len_total'] = np.sqrt(df['dx']**2 + df['dy']**2)

        # === DURATION SEGMENT ===
        df['segment_duration'] = df['dt']

        # === GEOMETRIC FEATURES (per file) ===
        width = df['x'].max() - df['x'].min()
        height = df['y'].max() - df['y'].min()

                # === GEOMETRIC FEATURES ===
        width = df['x'].max() - df['x'].min()
        height = df['y'].max() - df['y'].min()

        # === TOTAL PEN LIFT ===
        # penStatus = 0 artinya pena diangkat
        pen_lifts = ((df['penStatus'].shift(1) == 1) & (df['penStatus'] == 0)).sum()

        # === DIFFERENCE BETWEEN FIRST & LAST STROKE (Y) ===
        y_diff = df['y'].iloc[-1] - df['y'].iloc[0]

        # === VARIANCE OF Y POSITION ===
        var_y = df['y'].var()
        mean_y = df['y'].mean()
        median_y = df['y'].median()
        min_y = df['y'].min()
        max_y = df['y'].max()

        # === LOCAL EXTREMAS IN VELOCITY ===
        df['sign_v'] = np.sign(df['v_total'].diff())
        local_extremas_v = np.sum(df['sign_v'].diff().abs() == 2)

        # === LOCAL EXTREMAS IN ACCELERATION ===
        df['sign_a'] = np.sign(df['a_total'].diff())
        local_extremas_a = np.sum(df['sign_a'].diff().abs() == 2)

        # === TOTAL DURATION & LENGTH OF WRITING ===
        total_duration = df['time'].iloc[-1] - df['time'].iloc[0]
        total_length = df['segment_len_total'].sum()

        # === Hapus NaN/inf ===
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

         # --- Ambil info dari Excel ---
        match = df_info[df_info['filename'] == fn]
        if not match.empty:
            diag = match['diag'].values[0]
            age = match['age'].values[0]
            sex = match['sex'].values[0]
            hand = match['hand'].values[0]
        else:
            diag, age, sex, hand = None, None, None, None

        # === STATISTIK ===
        stats = {
            'filename': fn,
            'diag': diag,
            'age': age,
            'sex': sex,
            'hand': hand,

            # Velocity
            'mean_vx': df['v_x'].mean(),
            'median_vx': df['v_x'].median(),
            'std_vx': df['v_x'].std(),
            'min_vx': df['v_x'].min(),
            'max_vx': df['v_x'].max(),
            'p5_vx': np.percentile(df['v_x'], 5),
            'p95_vx': np.percentile(df['v_x'], 95),

            'mean_vy': df['v_y'].mean(),
            'median_vy': df['v_y'].median(),
            'std_vy': df['v_y'].std(),
            'min_vy': df['v_y'].min(),
            'max_vy': df['v_y'].max(),
            'p5_vy': np.percentile(df['v_y'], 5),
            'p95_vy': np.percentile(df['v_y'], 95),

            'mean_vtotal': df['v_total'].mean(),
            'median_vtotal': df['v_total'].median(),
            'std_vtotal': df['v_total'].std(),
            'min_vtotal': df['v_total'].min(),
            'max_vtotal': df['v_total'].max(),
            'p5_vtotal': np.percentile(df['v_total'], 5),
            'p95_vtotal': np.percentile(df['v_total'], 95),

            # Acceleration
            'mean_ax': df['a_x'].mean(),
            'median_ax': df['a_x'].median(),
            'std_ax': df['a_x'].std(),
            'min_ax': df['a_x'].min(),
            'max_ax': df['a_x'].max(),
            'p5_ax': np.percentile(df['a_x'], 5),
            'p95_ax': np.percentile(df['a_x'], 95),

            'mean_ay': df['a_y'].mean(),
            'median_ay': df['a_y'].median(),
            'std_ay': df['a_y'].std(),
            'min_ay': df['a_y'].min(),
            'max_ay': df['a_y'].max(),
            'p5_ay': np.percentile(df['a_y'], 5),
            'p95_ay': np.percentile(df['a_y'], 95),

            'mean_atotal': df['a_total'].mean(),
            'median_atotal': df['a_total'].median(),
            'std_atotal': df['a_total'].std(),
            'min_atotal': df['a_total'].min(),
            'max_atotal': df['a_total'].max(),
            'p5_atotal': np.percentile(df['a_total'], 5),
            'p95_atotal': np.percentile(df['a_total'], 95),

            # Jerk
            'mean_jx': df['j_x'].mean(),
            'median_jx': df['j_x'].median(),
            'std_jx': df['j_x'].std(),
            'min_jx': df['j_x'].min(),
            'max_jx': df['j_x'].max(),
            'p5_jx': np.percentile(df['j_x'], 5),
            'p95_jx': np.percentile(df['j_x'], 95),

            'mean_jy': df['j_y'].mean(),
            'median_jy': df['j_y'].median(),
            'std_jy': df['j_y'].std(),
            'min_jy': df['j_y'].min(),
            'max_jy': df['j_y'].max(),
            'p5_jy': np.percentile(df['j_y'], 5),
            'p95_jy': np.percentile(df['j_y'], 95),

            'mean_jtotal': df['j_total'].mean(),
            'median_jtotal': df['j_total'].median(),
            'std_jtotal': df['j_total'].std(),
            'min_jtotal': df['j_total'].min(),
            'max_jtotal': df['j_total'].max(),
            'p5_jtotal': np.percentile(df['j_total'], 5),
            'p95_jtotal': np.percentile(df['j_total'], 95),

            # === Length & Duration Segment ===
            'mean_lenx': df['segment_len_x'].mean(),
            'median_lenx': df['segment_len_x'].median(),
            'std_lenx': df['segment_len_x'].std(),
            'min_lenx': df['segment_len_x'].min(),
            'max_lenx': df['segment_len_x'].max(),

            'mean_leny': df['segment_len_y'].mean(),
            'median_leny': df['segment_len_y'].median(),
            'std_leny': df['segment_len_y'].std(),
            'min_leny': df['segment_len_y'].min(),
            'max_leny': df['segment_len_y'].max(),

            'mean_lentotal': df['segment_len_total'].mean(),
            'median_lentotal': df['segment_len_total'].median(),
            'std_lentotal': df['segment_len_total'].std(),
            'min_lentotal': df['segment_len_total'].min(),
            'max_lentotal': df['segment_len_total'].max(),

            'mean_duration': df['segment_duration'].mean(),
            'median_duration': df['segment_duration'].median(),
            'std_duration': df['segment_duration'].std(),
            'min_duration': df['segment_duration'].min(),
            'max_duration': df['segment_duration'].max(),

            # === Geometric shape ===
            'width': width,
            'height': height,

            # === Pressure, Altitude, Azimuth ===
            'mean_pressure': df['pressure'].mean(),
            'std_pressure': df['pressure'].std(),
            'min_pressure': df['pressure'].min(),
            'max_pressure': df['pressure'].max(),

            'mean_altitude': df['altitude'].mean(),
            'std_altitude': df['altitude'].std(),
            'min_altitude': df['altitude'].min(),
            'max_altitude': df['altitude'].max(),

            'mean_azimuth': df['azimuth'].mean(),
            'std_azimuth': df['azimuth'].std(),
            'min_azimuth': df['azimuth'].min(),
            'max_azimuth': df['azimuth'].max(),

            'y_diff_first_last': y_diff,
            'var_y': var_y,
            'mean_y': mean_y,
            'median_y': median_y,
            'min_y': min_y,
            'max_y': max_y,

            'pen_lift_count': pen_lifts,
            'local_extremas_velocity': local_extremas_v,
            'local_extremas_acceleration': local_extremas_a,
            'total_duration': total_duration,
            'total_length': total_length,
            'total_pen_lift': pen_lifts,  # disamakan untuk konsistensi

            'count': len(df)
        }

        summary_stats.append(stats)
        print(f"✅ {fn} selesai ({len(df)} sampel)")

    except Exception as e:
        print(f"⚠️ Gagal proses {fn}: {e}")

summary_df = pd.DataFrame(summary_stats)
display(summary_df)

out_csv = os.path.join(folder_path, 'onlineHWfeature.csv')
summary_df.to_csv(out_csv, index=False)
print(f"\n💾 Disimpan ke: {out_csv}")
