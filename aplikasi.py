import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

UNIT_LABEL = 'TON GABAH KERING GILING'
ALL_KEY = 'SEMUA PROVINSI'

DATA_PATH = Path('cleaned_produksi_padi_semua_kabupaten.csv')  # unified cleaned dataset

@st.cache_data(show_spinner=True)
def load_data(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"File {path} tidak ditemukan. Pastikan telah menjalankan notebook untuk menghasilkan file cleaned.")
	# Expect columns: nama_provinsi, tahun, bulan, date_parsed, produksi_padi
	df = pd.read_csv(path, parse_dates=['date_parsed'])
	# Standardize column names
	df.columns = [c.lower() for c in df.columns]
	# Backward compatibility if column uses nama_kabupaten_kota
	if 'nama_provinsi' not in df.columns and 'nama_kabupaten_kota' in df.columns:
		df = df.rename(columns={'nama_kabupaten_kota': 'nama_provinsi'})
	return df

def infer_seasonal_period(freq_str: str, series: pd.Series) -> int:
	if freq_str and ('M' in freq_str or freq_str in ['MS','M']):
		# Only return 12 if enough data points
		return 12 if len(series) >= 24 else None
	if freq_str and 'W' in freq_str:
		return 52 if len(series) >= 104 else None
	if freq_str and 'D' in freq_str:
		return 365 if len(series) >= 730 else None
	# fallback monthly assumption if length permits
	return 12 if len(series) >= 24 else None

def build_series(df: pd.DataFrame, region: str) -> pd.Series:
	if region == ALL_KEY:
		agg = (df.groupby('date_parsed', as_index=False)['produksi_padi'].sum()
				.sort_values('date_parsed'))
		s = agg.set_index('date_parsed')['produksi_padi'].astype(float)
	else:
		sub = df[df['nama_provinsi'] == region].copy()
		if sub.empty:
			raise ValueError(f"Data untuk {region} kosong.")
		sub = sub.sort_values('date_parsed')
		s = sub.set_index('date_parsed')['produksi_padi'].astype(float)
	inferred = pd.infer_freq(s.index)
	if inferred is None:
		s = s.resample('MS').sum().interpolate(method='linear')
		freq = 'MS'
	else:
		freq = inferred
	return s, freq

def ensure_strictly_positive(s: pd.Series, eps: float | None = None) -> tuple[pd.Series, float]:
	"""Replace non-positive values with a tiny epsilon suitable for multiplicative models."""
	s = s.astype(float).copy()
	if eps is None:
		pos = s[s > 0]
		base = float(pos.median()) if len(pos) > 0 else 1.0
		eps = max(1e-6, base * 1e-6)
	s[s <= 0] = eps
	return s, eps

def holt_winters_forecast(series: pd.Series, freq: str, horizon: int):
	# Ensure strictly positive inputs for multiplicative components
	series, _eps = ensure_strictly_positive(series)
	seasonal_periods = infer_seasonal_period(freq, series)
	# Force multiplicative seasonality; ensure seasonal_periods is set (default 12 for monthly)
	if seasonal_periods is None:
		seasonal_periods = 12
	trend = 'add'
	seasonal = 'mul'
	# holdout size: min(horizon, 12) but not exceeding data length
	h_test = min(horizon, 12, max(0, len(series)//10))
	train = series.iloc[:-h_test] if h_test > 0 else series
	test = series.iloc[-h_test:] if h_test > 0 else pd.Series(dtype=float)
	model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods, initialization_method='estimated')
	fit = model.fit(optimized=True)
	fitted = fit.fittedvalues
	# forecast test length (for metrics) then extend to cover test + future horizon
	forecast_test = fit.forecast(h_test) if h_test > 0 else pd.Series(dtype=float)
	# We need forecasts up to 'h_test' (for holdout) plus 'horizon' months beyond last_date
	future_h = h_test + max(horizon, 0)
	forecast_future = fit.forecast(future_h)

	def mape(y_true, y_pred):
		y_true, y_pred = np.array(y_true), np.array(y_pred)
		mask = y_true != 0
		if mask.sum() == 0:
			return np.nan
		return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

	metrics = None
	if h_test > 0 and len(test) == h_test:
		metrics = {
			'MAE': float(mean_absolute_error(test, forecast_test)),
			'RMSE': float(np.sqrt(mean_squared_error(test, forecast_test))),
			'MAPE': float(mape(test, forecast_test))
		}
	return {
		'fit': fit,
		'fitted': fitted,
		'forecast_test': forecast_test,
		'test': test,
		'forecast_future': forecast_future,
		'metrics': metrics,
		'seasonal_periods': seasonal_periods,
		'trend': trend,
		'seasonal': seasonal
	}

def plot_forecast(series: pd.Series, fitted: pd.Series, test: pd.Series, forecast_future: pd.Series, horizon: int):
	fig, ax = plt.subplots(figsize=(10,4))
	ax.plot(series.index, series.values, label='Actual', color='black')
	ax.plot(fitted.index, fitted.values, label='Fitted', color='tab:blue')
	if len(test) > 0:
		ax.plot(test.index, forecast_future.iloc[:len(test)].values, label='Forecast (Holdout)', color='tab:orange')
		# future beyond test
		extra = forecast_future.iloc[len(test):]
		if len(extra) > 0:
			future_index = pd.date_range(series.index.max() + pd.offsets.MonthBegin(1), periods=len(extra), freq='MS')
			ax.plot(future_index, extra.values, label='Future', color='tab:green')
	else:
		future_index = pd.date_range(series.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq='MS')
		ax.plot(future_index, forecast_future.values, label='Forecast', color='tab:orange')
	ax.set_title('Holt-Winters Forecast')
	ax.set_xlabel('Tanggal')
	ax.set_ylabel('Produksi Padi')
	ax.legend()
	ax.grid(alpha=0.2)
	st.pyplot(fig)

def main():
	st.set_page_config(page_title='Forecast Produksi Padi Jabar', layout='wide')
	st.title('Forecasting Produksi Padi Jawa Barat')
	df = load_data(DATA_PATH)
	regions = [ALL_KEY] + sorted(df['nama_provinsi'].unique())

	col_sel, col_date, col_info = st.columns([2,2,1])
	with col_sel:
		region = st.selectbox('Pilih Kabupaten/Kota', regions, index=0)
	with col_date:
		# Input target forecast month/year
		years = list(range(df['tahun'].min(), df['tahun'].max()+4))  # allow up to +5 years ahead
		months_map = {1:'Januari',2:'Februari',3:'Maret',4:'April',5:'Mei',6:'Juni',7:'Juli',8:'Agustus',9:'September',10:'Oktober',11:'November',12:'Desember'}
		month_names = [months_map[m] for m in range(1,13)]
		col_m, col_y = st.columns(2)
		with col_m:
			month_name = st.selectbox('Bulan Target', month_names, index=11)  # default Desember
		with col_y:
			year_target = st.selectbox('Tahun Target', years, index=len(years)-1)
		month_target = [k for k,v in months_map.items() if v==month_name][0]
		# We'll compute horizon dynamically later

	with st.expander('Lihat Data Asli (Subset)', expanded=False):
		st.dataframe(df[df['nama_provinsi'] == region].sort_values('date_parsed').head(50))

	series, freq = build_series(df, region)
	last_date = series.index.max()
	# Target date is first day of target month
	target_date = pd.Timestamp(year_target, month_target, 1)
	# Compute horizon (months) difference
	if target_date <= last_date:
		st.warning(f'Tanggal target {target_date.date()} berada dalam atau sebelum data historis. Menampilkan nilai aktual.')
		computed_horizon = 0
	else:
		computed_horizon = (target_date.year - last_date.year) * 12 + (target_date.month - last_date.month)
		st.info(f'Horizon otomatis: {computed_horizon} bulan dari {last_date.date()} ke {target_date.date()}')
	# Cap maximum horizon
	max_allowed = 36
	if computed_horizon > max_allowed:
		st.error(f'Horizon {computed_horizon} bulan melebihi batas {max_allowed}. Silakan pilih bulan lebih dekat.')
		return
	# Define horizon used for modeling (at least 1 to allow forecasting machinery)
	horizon_for_model = max(computed_horizon, 1)
	result = holt_winters_forecast(series, freq, horizon_for_model)

	st.subheader('Ringkasan Model')
	st.write({
		'Seasonal Periods': result['seasonal_periods'],
		'Trend Component': result['trend'],
		'Seasonal Component': result['seasonal'],
		'Data Points': len(series),
		'Frequency': freq
	})

	if result['metrics']:
		st.subheader('Metrix Holdout')
		m_df = pd.DataFrame([result['metrics']])
		st.table(m_df)
	else:
		st.info('Tidak cukup data untuk evaluasi holdout.')

	st.subheader('Plot Forecast')
	# Use computed horizon for plotting future axis (even if model horizon forced to 1)
	plot_forecast(series, result['fitted'], result['test'], result['forecast_future'], computed_horizon)

	st.subheader('Nilai Bulan Target')
	if computed_horizon > 0:
		# Ambil hanya nilai untuk bulan target (bagian setelah holdout)
		future_index_full = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=computed_horizon, freq='MS')
		# Ambil bagian forecast setelah segmen holdout
		forecast_extra = pd.Series(result['forecast_future']).iloc[len(result['test']):]
		# Pastikan panjang sesuai computed_horizon
		forecast_extra = forecast_extra.iloc[:computed_horizon]
		forecast_full = pd.DataFrame({'tanggal': future_index_full, 'forecast': forecast_extra.values})
		value_target = forecast_full.loc[forecast_full['tanggal'] == target_date, 'forecast']
		if not value_target.empty:
			val = value_target.values[0]
			label_region = 'Provinsi Total' if region == ALL_KEY else region
			file_region = 'total_provinsi' if region == ALL_KEY else region
			st.metric(label=f'Forecast {label_region} {month_name} {year_target}', value=f"{val:,.2f} {UNIT_LABEL}")
			st.download_button('Unduh Forecast Bulan Target (CSV)', data=forecast_full.loc[forecast_full['tanggal']==target_date].to_csv(index=False), file_name=f'forecast_{file_region}_{target_date.year}_{target_date.month}.csv', mime='text/csv')
			st.subheader('Forecast Hingga Bulan Target')
			st.dataframe(forecast_full)
			st.download_button('Unduh Semua Forecast Hingga Target (CSV)', data=forecast_full.to_csv(index=False), file_name=f'forecast_range_{file_region}_{last_date.year}_{last_date.month}_to_{target_date.year}_{target_date.month}.csv', mime='text/csv')
		else:
			st.warning('Forecast target tidak ditemukan (periksa horizon).')
	else:
		# target berada di historis -> tampilkan nilai aktual
		if target_date in series.index:
			actual_val = series.loc[target_date]
			st.metric(label=f'Nilai Aktual {region} {month_name} {year_target}', value=f"{actual_val:,.2f} {UNIT_LABEL}")
		else:
			st.warning('Nilai aktual untuk tanggal target tidak tersedia.')

	st.markdown('---')
if __name__ == '__main__':
	main()

