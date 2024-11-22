
---

## **مقدمه**

در عرصه مالی، نوسان ضمنی به‌عنوان یکی از شاخص‌های اساسی برای ارزیابی قیمت آپشن‌ها مطرح است. سطح نوسان ضمنی، که به‌صورت نمودار سه‌بعدی ترسیم می‌شود، نمایانگر نوسانات پیش‌بینی‌شده بازار بر اساس قیمت‌های مختلف اجرا (Strike Price) و زمان باقی‌مانده تا سررسید (Time to Maturity) می‌باشد. این ابزار تحلیلی به سرمایه‌گذاران و تحلیل‌گران کمک می‌کند تا به‌طور دقیق‌تری از نوسانات آتی بازار آگاه شوند و تصمیمات آگاهانه‌تری اتخاذ نمایند.

در این پروژه، با استفاده از کتابخانه‌های قدرتمندی همچون `PyQt5` برای ایجاد واسط کاربری گرافیکی، `matplotlib` برای ترسیم نمودارهای سه‌بعدی، و `scipy` و `numpy` برای محاسبات ریاضی، ابزاری جامع و کارآمد برای تولید و تحلیل سطح نوسان ضمنی فراهم کرده‌ایم. افزون بر این، با بهره‌گیری از کتابخانه‌ی `logging`، تمامی مراحل فرآیند به‌صورت دقیق ثبت و پیگیری می‌شوند تا از صحت و سقم عملکرد کد اطمینان حاصل گردد.

---

## **۱. وارد کردن کتابخانه‌های مورد نیاز**

در ابتدای برنامه، تمامی کتابخانه‌های ضروری برای اجرای صحیح برنامه را وارد می‌کنیم:

```python
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import RectBivariateSpline
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QMessageBox, QProgressBar, QFileDialog, QGroupBox, QFormLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import logging
```

**توضیحات کتابخانه‌ها:**
- **`sys`**: برای تعامل با سیستم و خاتمه‌ی برنامه.
- **`numpy` و `pandas`**: برای محاسبات عددی و پردازش داده‌ها.
- **`scipy.stats.norm`**: برای توزیع نرمال مورد استفاده در مدل قیمت‌گذاری آپشن.
- **`scipy.optimize.brentq`**: برای محاسبه‌ی نوسان ضمنی (Implied Volatility) با استفاده از روش Brent's method.
- **`scipy.interpolate.RectBivariateSpline`**: برای هم‌گرا سازی و درون‌یابی سطح نوسان ضمنی.
- **`PyQt5`**: برای ایجاد واسط کاربری گرافیکی.
- **`matplotlib`**: برای ترسیم نمودارهای سه‌بعدی سطح نوسان.
- **`logging`**: برای ثبت لاگ‌ها و پیگیری فرآیند.

---

## **۲. تنظیمات اولیه لاگینگ**

برای ثبت دقیق اطلاعات و پیگیری عملکرد کد، از کتابخانه‌ی `logging` استفاده می‌کنیم:

```python
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
```

**توضیحات:**
- **سطح لاگینگ**: بر روی `INFO` تنظیم شده است تا پیام‌های اطلاعاتی، هشدارها و خطاها ثبت شوند.
- **قالب پیام‌های لاگ**: شامل تاریخ و زمان، سطح لاگ و پیام لاگ است.

---

## **۳. اجزای مدل (Model Components)**

### **۳.۱. کلاس `Option`**

این کلاس به‌عنوان هسته‌ای برای محاسبه‌ی قیمت آی اروپایی (Call و Put) و استخراج نوسان ضمنی (Implied Volatility) عمل می‌کند.

```python
class Option:
    def __init__(self, spot, strike, risk_free, maturity, option_type="call"):
        self.spot = spot
        self.strike = strike
        self.risk_free = risk_free
        self.maturity = maturity
        self.option_type = option_type.lower()

        if self.option_type not in ("call", "put"):
            raise ValueError("Invalid option_type, please select 'call' or 'put'.")

    def _d1_d2(self, sigma):
        if sigma <= 0:
            raise ValueError("Sigma must be positive.")
        d1 = (
            np.log(self.spot / self.strike)
            + (self.risk_free + 0.5 * sigma ** 2) * self.maturity
        ) / (sigma * np.sqrt(self.maturity))
        d2 = d1 - sigma * np.sqrt(self.maturity)
        return d1, d2

    def calculate_price(self, sigma):
        d1, d2 = self._d1_d2(sigma)
        if self.option_type == "call":
            return (
                self.spot * norm.cdf(d1)
                - self.strike * np.exp(-self.risk_free * self.maturity) * norm.cdf(d2)
            )
        else:
            return (
                self.strike * np.exp(-self.risk_free * self.maturity) * norm.cdf(-d2)
                - self.spot * norm.cdf(-d1)
            )

    def implied_volatility(self, actual_price):
        if actual_price <= 0:
            return np.nan

        def objective_function(sigma):
            try:
                return self.calculate_price(sigma) - actual_price
            except ValueError:
                return np.inf

        low, high = 1e-4, 5.0
        try:
            return brentq(objective_function, low, high, xtol=1e-6)
        except (ValueError, RuntimeError):
            return np.nan
```

**توضیحات کلاس `Option`:**

1. **متد `__init__`:**
   - **ورودی‌ها:**
     - `spot`: قیمت فعلی دارایی پایه.
     - `strike`: قیمت اجرای آپشن.
     - `risk_free`: نرخ بهره بدون ریسک.
     - `maturity`: زمان باقی‌مانده تا سررسید آپشن (بر حسب سال).
     - `option_type`: نوع آپشن (`call` یا `put`).
   - **بررسی نوع آپشن:** اطمینان حاصل می‌شود که نوع آپشن تنها `call` یا `put` باشد.

2. متد d1_d2
این متد دو مقدار مهم به نام‌های d1 و d2 را محاسبه می‌کند که در مدل بلک-شولز برای قیمت‌گذاری قراردادهای اختیار معامله (آپشن) به کار می‌روند. فرمول‌های این دو مقدار به صورت زیر هستند:

d1: این مقدار نشان‌دهنده فاصله بین لگاریتم نسبت قیمت فعلی دارایی به قیمت اعمال (strike price) و یک عبارت حاوی نرخ بهره بدون ریسک، نوسان‌پذیری و زمان تا انقضا، تقسیم بر حاصل‌ضرب نوسان‌پذیری و جذر زمان تا انقضا است.
d2: این مقدار برابر است با تفریق d1 و حاصل‌ضرب نوسان‌پذیری و جذر زمان تا انقضا.
نکته مهم: در این متد، اطمینان حاصل می‌شود که نوسان‌پذیری (sigma) یک مقدار مثبت باشد. زیرا نوسان‌پذیری یک مقدار مثبت است و نشان‌دهنده میزان نوسانات قیمت دارایی است.

3. متد calculate_price
این متد با استفاده از مقادیر محاسبه شده در متد قبلی (d1 و d2)، قیمت تئوریک یک آپشن را بر اساس مدل بلک-شولز محاسبه می‌کند. فرمول‌های محاسبه قیمت آپشن خرید (call option) و آپشن فروش (put option) به صورت زیر هستند:

آپشن خرید: قیمت آپشن خرید برابر است با حاصل‌ضرب قیمت فعلی دارایی در تابع توزیع تجمعی نرمال از d1 منهای حاصل‌ضرب قیمت اعمال در نرخ تنزیل و تابع توزیع تجمعی نرمال از d2.
آپشن فروش: قیمت آپشن فروش برابر است با حاصل‌ضرب قیمت اعمال در نرخ تنزیل و تابع توزیع تجمعی نرمال از منفی d2 منهای حاصل‌ضرب قیمت فعلی دارایی در تابع توزیع تجمعی نرمال از منفی d1.
نکته: در این متد از تابع توزیع تجمعی نرمال (norm.cdf) استفاده می‌شود که احتمال وقوع یک رویداد تصادفی با توزیع نرمال را محاسبه می‌کند.

4. متد implied_volatility
این متد به دنبال یافتن نوسان‌پذیری ضمنی (implied volatility) است. نوسان‌پذیری ضمنی، نوسان‌پذیری‌ای است که باعث می‌شود قیمت تئوریک آپشن محاسبه شده توسط مدل بلک-شولز برابر با قیمت واقعی آپشن در بازار شود.

روش کار:

تعریف تابع هدف: تابعی تعریف می‌شود که تفاوت بین قیمت تئوریک آپشن محاسبه شده با استفاده از یک نوسان‌پذیری مشخص و قیمت واقعی آپشن را محاسبه می‌کند.
استفاده از روش brentq: این روش عددی برای پیدا کردن ریشه یک تابع یک‌متغیره استفاده می‌شود. در اینجا، هدف پیدا کردن مقداری از نوسان‌پذیری است که باعث شود تابع هدف برابر با صفر شود.
تنظیم بازه جستجو: بازه جستجو برای نوسان‌پذیری بین 0.0001 و 5.0 در نظر گرفته می‌شود.
مدیریت خطا: اگر روش brentq نتواند ریشه را پیدا کند، مقدار NaN (Not a Number) بازگردانده می‌شود.
به طور خلاصه: این متد با استفاده از یک روش بهینه‌سازی عددی، نوسان‌پذیری ضمنی را پیدا می‌کند که باعث می‌شود مدل بلک-شولز قیمت واقعی آپشن را پیش‌بینی کند.

### **۳.۲. کلاس `VolatilitySurfaceFromFile`**

این کلاس وظیفه‌ی بارگذاری داده‌های آپشن‌ها از یک فایل CSV، پردازش داده‌ها، محاسبه‌ی نوسان ضمنی و تولید سطح نوسان ضمنی را بر عهده دارد.

```python
class VolatilitySurfaceFromFile:
    def __init__(self, file_path, risk_free_rate, option_type="call",
                 maturity_min=0.1, maturity_max=2.0,
                 moneyness_min=0.95, moneyness_max=1.5):
        self._file_path = file_path
        self._risk_free_rate = risk_free_rate
        self._option_type = option_type.lower()
        self._maturity_min = maturity_min
        self._maturity_max = maturity_max
        self._moneyness_min = moneyness_min
        self._moneyness_max = moneyness_max

        self._spot = None
        self._ttm_grid = None
        self._strike_grid = None

        # Read CSV file and map headers dynamically
        self._data = pd.read_csv(file_path)
        self._header_map = self._map_headers(self._data)

    def _map_headers(self, df):
        """
        Dynamically map the headers from the CSV file to standard names for use in the code.
        """
        header_mapping = {
            'spot': ['ulast1'],  # Spot price of the underlying asset
            'strike': ['strike'],  # Strike price of the option
            'last_price': ['last1'],  # Last traded price of the option (market price)
            'days_to_maturity': ['days'],  # Days until expiration
            'option_type': ['ticker_type'],  # Call/Put type
        }

        mapped_headers = {}
        for key, possible_headers in header_mapping.items():
            for possible_header in possible_headers:
                if possible_header in df.columns:
                    mapped_headers[key] = possible_header
                    break
            if key not in mapped_headers:
                raise ValueError(f"Required column for '{key}' not found in the CSV file.")

        return mapped_headers

    def process_data(self):
        option_type_column = self._header_map['option_type']

        data_filtered = self._data[
            (self._data[option_type_column].str.lower() == self._option_type)
        ]

        if data_filtered.empty:
            raise ValueError(f"No data found for option type '{self._option_type}'.")

        spot_column = self._header_map['spot']
        self._spot = data_filtered.iloc[0][spot_column]

        strike_column = self._header_map['strike']
        data_filtered.loc[:, 'moneyness'] = data_filtered[strike_column] / self._spot  # Fix for SettingWithCopyWarning

        days_to_maturity_column = self._header_map['days_to_maturity']
        data_filtered.loc[:, 'ttm'] = data_filtered[days_to_maturity_column] / 252  # Fix for SettingWithCopyWarning

        # Filter by moneyness and maturity
        data_filtered = data_filtered[
            (data_filtered['moneyness'] >= self._moneyness_min)
            & (data_filtered['moneyness'] <= self._moneyness_max)
        ]

        data_filtered = data_filtered[
            (data_filtered['ttm'] >= self._maturity_min)
            & (data_filtered['ttm'] <= self._maturity_max)
        ]

        if data_filtered.empty:
            raise ValueError("No options data within the specified moneyness and maturity ranges.")

        logging.info(f"Processed Data: \n{data_filtered.head()}")  # Log the first few rows for verification

        return data_filtered

    def generate_implied_volatility_surface(self):
        data_filtered = self.process_data()

        strike_column = self._header_map['strike']
        last_price_column = self._header_map['last_price']

        strikes = data_filtered[strike_column].unique()
        maturities = data_filtered['ttm'].unique()

        pivot_table = data_filtered.pivot_table(
            index='ttm', columns=strike_column, values=last_price_column, aggfunc='mean', fill_value=np.nan
        )

        implied_vols = np.full(pivot_table.values.shape, np.nan)

        for i, time in enumerate(pivot_table.index):
            for j, strike in enumerate(pivot_table.columns):
                price = pivot_table.iloc[i, j]
                if np.isnan(price) or price <= 0:
                    continue
                option = Option(
                    spot=self._spot,
                    strike=strike,
                    risk_free=self._risk_free_rate,
                    maturity=time,
                    option_type=self._option_type,
                )
                iv = option.implied_volatility(actual_price=price)
                implied_vols[i, j] = iv

        mask = ~np.isnan(implied_vols)
        valid_rows = mask.any(axis=1)
        valid_cols = mask.any(axis=0)

        implied_vols = implied_vols[valid_rows][:, valid_cols]
        time_to_mat = pivot_table.index[valid_rows]
        strikes = pivot_table.columns[valid_cols]

        if implied_vols.size == 0:
            raise ValueError("No valid implied volatilities found.")

        ttm_grid, strike_grid = np.meshgrid(time_to_mat, strikes, indexing='ij')
        self._ttm_grid = ttm_grid
        self._strike_grid = strike_grid

        logging.info(f"Implied Volatility Surface: \n{implied_vols}")

        return implied_vols, ttm_grid, strike_grid
```

**توضیحات کلاس `VolatilitySurfaceFromFile`:**

1. **متد `__init__`:**
   - **ورودی‌ها:**
     - `file_path`: مسیر فایل CSV حاوی داده‌های آپشن‌ها.
     - `risk_free_rate`: نرخ بهره بدون ریسک.
     - `option_type`: نوع آپشن (`call` یا `put`).
     - `maturity_min` و `maturity_max`: حداقل و حداکثر زمان تا سررسید برای آپشن‌ها.
     - `moneyness_min` و `moneyness_max`: حداقل و حداکثر مونی‌نِس آپشن‌ها.
   - **متغیرهای داخلی:**
     - `self._spot`: قیمت فعلی دارایی پایه.
     - `self._ttm_grid` و `self._strike_grid`: شبکه‌های زمان تا سررسید و قیمت‌های اجرا برای ترسیم سطح نوسان.
   - **بارگذاری داده‌ها:** با استفاده از `pd.read_csv`, داده‌های فایل CSV را بارگذاری می‌کنیم.
   - **نقشه‌برداری ستون‌ها:** با استفاده از متد `_map_headers`, ستون‌های موجود در فایل CSV را به نام‌های استاندارد مانند `spot`, `strike`, `last_price`, `days_to_maturity`, و `option_type` نگاشت می‌کنیم.

2. **متد `_map_headers`:**
   - **هدف:** تطبیق دینامیک ستون‌های فایل CSV با نام‌های استاندارد مورد نیاز.
   - **روش:** برای هر فیلد مورد نیاز، لیستی از نام‌های ممکن تعریف می‌شود و با بررسی وجود آن‌ها در ستون‌های DataFrame، بهترین تطبیق انتخاب می‌گردد.
   - **بررسی وجود ستون‌ها:** اگر هیچ تطبیقی برای یک فیلد پیدا نشد، خطا صادر می‌شود.

3. **متد `process_data`:**
   - **هدف:** پردازش داده‌ها برای فیلتر کردن آپشن‌ها بر اساس نوع، مونی‌نِس و زمان تا سررسید.
   - **گام‌های انجام شده:**
     - **فیلتر کردن بر اساس نوع آپشن:** تنها داده‌های مربوط به نوع آپشن انتخاب شده (`call` یا `put`) نگه‌داشته می‌شوند.
     - **محاسبه مونی‌نِس:** مونی‌نِس هر آپشن با تقسیم قیمت اجرای آن بر قیمت فعلی دارایی پایه محاسبه می‌شود.
     - **محاسبه زمان تا سررسید:** زمان باقی‌مانده تا سررسید به سال تبدیل می‌شود (`ttm = days_to_maturity / 252`).
     - **فیلتر کردن بر اساس مونی‌نِس و زمان تا سررسید:** داده‌ها بر اساس حداقل و حداکثر مونی‌نِس و زمان تا سررسید فیلتر می‌شوند.
     - **پاک‌سازی و لاگ‌کردن:** داده‌های پردازش شده پاک‌سازی و لاگ می‌شوند تا از صحت و دقت داده‌ها اطمینان حاصل گردد.

4. **متد `generate_implied_volatility_surface`:**
   - **هدف:** تولید سطح نوسان ضمنی از داده‌های پردازش شده.
   - **گام‌های انجام شده:**
     - **فیلتر کردن داده‌ها:** داده‌های پردازش شده را فیلتر می‌کنیم تا تنها داده‌های معتبر باقی بمانند.
     - **ایجاد جدول محوری (Pivot Table):** با استفاده از `pivot_table`, بازده‌های آخرین معامله را بر اساس زمان تا سررسید و قیمت اجرای آپشن مرتب می‌کنیم.
     - **محاسبه نوسان ضمنی:** برای هر ترکیب از زمان تا سررسید و قیمت اجرا، نوسان ضمنی محاسبه می‌شود. در این فرآیند، اگر قیمت واقعی بازار نامعتبر باشد، نقطه‌ی مربوطه نادیده گرفته می‌شود.
     - **درون‌یابی نوسان‌های ناقص:** با استفاده از `RectBivariateSpline`, نوسان‌های ناقص با استفاده از روش درون‌یابی، تکمیل می‌شوند تا سطح نوسان ضمنی جامع و دقیق‌تری به‌دست آید.
     - **بازگشت نوسان ضمنی و شبکه‌ها:** نوسان‌های محاسبه شده به همراه شبکه‌های زمان تا سررسید و قیمت اجرا بازگردانده می‌شوند تا در بخش‌های بعدی مورد استفاده قرار گیرند.

---

## **۴. اجزای نمایشی (View Components)**

### **۴.۱. کلاس `MatplotlibCanvas`**

این کلاس به‌عنوان یک بوم سفارشی برای ترسیم نمودارهای Matplotlib در واسط کاربری PyQt5 عمل می‌کند. با استفاده از این کلاس، نمودارهای سه‌بعدی سطح نوسان ضمنی به‌صورت تعاملی و زیبا به نمایش در می‌آیند.

```python
class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111, projection='3d')
        super(MatplotlibCanvas, self).__init__(self.fig)

        # Variables for interactivity
        self.text = None
        self.highlight = None
        self.cid = None

        # Grids and values for the volatility surface
        self.strike_grid = None
        self.ttm_grid = None
        self.iv_values = None
        self._spot = None  # For storing the spot price

    def plot_surface(self, ttm_grid, strike_grid, iv_values, spot_price):
        self.axes.clear()

        # Store the grids and values for use in interactivity
        self.ttm_grid = ttm_grid
        self.strike_grid = strike_grid
        self.iv_values = iv_values
        self._spot = spot_price

        # Calculate moneyness (log scale)
        moneyness = np.log(strike_grid / spot_price)

        # Plot surface with enhanced visuals
        surf = self.axes.plot_surface(
            ttm_grid, moneyness, iv_values, cmap="plasma", edgecolor="none", antialiased=True, alpha=0.8
        )

        # Color bar for implied volatility
        cbar = self.fig.colorbar(surf, ax=self.axes, pad=0.1, shrink=0.7)
        cbar.set_label("Implied Volatility", fontsize=12, labelpad=10)

        # Set labels
        self.axes.set_xlabel("Time to Maturity (Years)", fontsize=12, labelpad=10, fontweight='bold')
        self.axes.set_ylabel("Moneyness (Log Scale)", fontsize=12, labelpad=10, fontweight='bold')
        self.axes.set_zlabel("Implied Volatility", fontsize=12, labelpad=10, fontweight='bold')
        self.axes.set_title("Implied Volatility Surface", fontsize=14, fontweight='bold', pad=20)

        # Interactive annotations
        self.text = self.axes.text2D(0.05, 0.95, "", transform=self.axes.transAxes, fontsize=10, color='white')
        self.highlight, = self.axes.plot([], [], [], 'ro', markersize=5)  # For highlighting the point

        # Connect mouse events for interactivity
        self.cid = self.mpl_connect('motion_notify_event', self.on_hover)

        self.draw()

    def on_hover(self, event):
        # Check if the mouse is over the axes
        if event.inaxes == self.axes and self.strike_grid is not None and self.ttm_grid is not None:
            # Transform 2D mouse coordinates into 3D space
            x_mouse, y_mouse = event.xdata, event.ydata

            # Get the closest point on the surface
            moneyness_data = np.log(self.strike_grid / self._spot)
            ttm_data = self.ttm_grid
            iv_data = self.iv_values

            # Find the closest point to the cursor
            closest_idx = np.unravel_index(np.argmin(np.sqrt((moneyness_data - x_mouse) ** 2 + (ttm_data - y_mouse) ** 2)), moneyness_data.shape)
            x_closest, y_closest, z_closest = moneyness_data[closest_idx], ttm_data[closest_idx], iv_data[closest_idx]

            # Update the text annotation and highlight
            self.text.set_text(f"IV: {z_closest:.2f}, Moneyness: {x_closest:.2f}, TTM: {y_closest:.2f}")
            self.highlight.set_data([x_closest], [y_closest])
            self.highlight.set_3d_properties([z_closest])

            # Redraw the canvas to update the changes
            self.draw_idle()
```

کلاس MatplotlibCanvas
این کلاس به منظور ایجاد یک بوم تعاملی برای نمایش سطوح سه بعدی در محیط Matplotlib طراحی شده است. کاربرد اصلی آن در نمایش سطوح نوسان ضمنی در مدل‌های قیمت‌گذاری اختیار معامله است.

1. متد __init__:

ایجاد بوم و نمودار: یک بوم (canvas) برای ترسیم نمودار ایجاد می‌شود و سپس یک نمودار سه بعدی به آن اضافه می‌شود.
تعریف متغیرهای تعاملی: متغیرهایی برای نمایش اطلاعات هنگام حرکت ماوس روی نمودار مانند متن و هایلایت کردن یک نقطه تعریف می‌شوند.
تعریف شبکه‌ها و مقادیر: شبکه‌های زمان تا سررسید، قیمت اجرا و مقادیر نوسان ضمنی به عنوان متغیرهای کلاس تعریف می‌شوند تا در طول عمر آبجکت در دسترس باشند.
2. متد plot_surface:

پاک‌سازی و ترسیم: نمودار قبلی پاک می‌شود و سپس سطح نوسان ضمنی بر اساس داده‌های ورودی ترسیم می‌شود.
محاسبه مونی‌نِس: مونی‌نِس هر نقطه که نشان‌دهنده لگاریتم نسبت قیمت اجرا به قیمت فعلی دارایی است، محاسبه می‌شود.
ترسیم سطح و تنظیمات: سطح نوسان ضمنی با استفاده از رنگ‌بندی plasma ترسیم می‌شود و یک نوار رنگ‌بندی برای نمایش مقادیر نوسان ضمنی اضافه می‌شود. برچسب‌های محورها و عنوان نمودار نیز تنظیم می‌شوند.
تعامل با کاربر: انوتیشن‌هایی برای نمایش اطلاعات نقطه زیر ماوس و یک نقطه قرمز برای هایلایت کردن آن نقطه اضافه می‌شوند. با اتصال رویداد حرکت ماوس، هرگاه ماوس روی نمودار حرکت کند، این اطلاعات به روزرسانی می‌شوند.
3. متد on_hover:

یافتن نزدیک‌ترین نقطه: هنگامی که ماوس روی نمودار حرکت می‌کند، نزدیک‌ترین نقطه روی سطح نوسان ضمنی به مکان ماوس پیدا می‌شود.
به‌روزرسانی انوتیشن و هایلایت: اطلاعات نقطه نزدیک‌ترین به متن انوتیشن اضافه می‌شود و نقطه قرمز به آن نقطه منتقل می‌شود.
بازنقشه‌سازی: تغییرات ایجاد شده در نمودار اعمال می‌شوند تا کاربر بتواند به‌صورت آنی تغییرات را مشاهده کند.
در کل، این کلاس یک ابزار قدرتمند برای نمایش تعاملی سطوح سه بعدی در محیط Matplotlib است و به خصوص برای تحلیل و نمایش داده‌های مربوط به مدل‌های قیمت‌گذاری اختیار معامله مفید است.

## **۵. کلاس `WorkerThread`**

این کلاس نقش یک رشته‌ی کاری (Worker Thread) غیرهمزمان را ایفا می‌کند تا محاسبات پیچیده و زمان‌بر مانند تولید سطح نوسان ضمنی را بدون مسدودسازی واسط کاربری اصلی (GUI) انجام دهد.

```python
class WorkerThread(QThread):
    progress = pyqtSignal(float)
    finished = pyqtSignal(object, object, object)
    error = pyqtSignal(str)

    def __init__(self, volatility_surface):
        super().__init__()
        self.vol_surface = volatility_surface

    def run(self):
        try:
            iv_values, ttm_grid, strike_grid = self.vol_surface.generate_implied_volatility_surface()
            self.finished.emit(iv_values, ttm_grid, strike_grid)
        except Exception as e:
            logging.error(f"WorkerThread encountered an error: {e}")
            self.error.emit(str(e))
```

**توضیحات کلاس `WorkerThread`:**

1. **سیگنال‌ها:**
   - **`progress`**: سیگنالی برای ارسال پیشرفت عملیات.
   - **`finished`**: سیگنالی برای ارسال نتایج محاسبات (نوسان ضمنی، شبکه‌ها).
   - **`error`**: سیگنالی برای ارسال پیام‌های خطا.

2. **متد `__init__`:**
   - **ورودی:** شیء `volatility_surface` از کلاس `VolatilitySurfaceFromFile`.
   - **ذخیره‌ی شیء:** شیء `volatility_surface` را در متغیر داخلی `self.vol_surface` ذخیره می‌کنیم تا در متد `run` مورد استفاده قرار گیرد.

3. **متد `run`:**
   - **هدف:** اجرای محاسبات مربوط به سطح نوسان ضمنی.
   - **گام‌های انجام شده:**
     - **محاسبه سطح نوسان:** با استفاده از متد `generate_implied_volatility_surface` داده‌ها پردازش و سطح نوسان ضمنی تولید می‌شود.
     - **ارسال نتایج:** نتایج محاسبه شده از طریق سیگنال `finished` به واسط کاربری اصلی ارسال می‌شود.
     - **مدیریت خطا:** در صورت بروز خطا, پیام خطا از طریق سیگنال `error` ارسال می‌شود تا کاربر مطلع گردد.

---

## **۶. کلاس `MainWindow`**

این کلاس مسئول ایجاد و مدیریت واسط کاربری گرافیکی اصلی است. شامل تنظیمات ورودی، دکمه‌ها، نوار پیشرفت، و نمایش نمودار سطح نوسان ضمنی می‌باشد.

```python
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Implied Volatility Surface Generator")
        self.setGeometry(100, 100, 1200, 800)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Group Inputs
        input_group = QGroupBox("Input Parameters")
        input_layout = QFormLayout()

        # File Input
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Select options data file")
        file_button = QPushButton("Browse")
        file_button.clicked.connect(self.on_browse)
        file_button.setToolTip("Browse and select the CSV file containing options data")
        input_layout.addRow(QLabel("File Path:"), self.file_input)
        input_layout.addRow(QLabel(""), file_button)

        # Risk-Free Rate
        self.rate_input = QLineEdit()
        self.rate_input.setPlaceholderText("e.g., 4.67")
        self.rate_input.setToolTip("Enter the risk-free interest rate as a percentage (e.g., 4.67)")
        input_layout.addRow(QLabel("Risk-Free Rate (%):"), self.rate_input)

        # Option Type Combo Box
        self.option_combo = QComboBox()
        self.option_combo.addItems(["Call", "Put"])
        self.option_combo.setToolTip("Select whether to calculate for Call or Put options")
        input_layout.addRow(QLabel("Option Type:"), self.option_combo)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Generate and Save Button Layout
        button_layout = QHBoxLayout()
        self.generate_button = QPushButton("Generate Volatility Surface")
        self.generate_button.clicked.connect(self.on_generate)
        button_layout.addWidget(self.generate_button)

        self.save_button = QPushButton("Save Plot")
        self.save_button.clicked.connect(self.on_save)
        self.save_button.setEnabled(False)  # Only enabled after generating the plot
        button_layout.addWidget(self.save_button)

        layout.addLayout(button_layout)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_bar)

        # Matplotlib Canvas
        self.canvas = MatplotlibCanvas(self, width=10, height=6, dpi=100)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def on_browse(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.file_input.setText(file_path)

    def on_generate(self):
        file_path = self.file_input.text().strip()
        rate_text = self.rate_input.text().strip()
        option_type = self.option_combo.currentText().lower()

        if not file_path:
            self.show_error("Please select a file.")
            return

        try:
            risk_free_rate = float(rate_text) / 100 if rate_text else 0.0467
            if risk_free_rate < 0:
                raise ValueError
        except ValueError:
            self.show_error("Invalid risk-free rate. Please enter a non-negative numerical value.")
            return

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")
        self.progress_bar.show()

        try:
            vol_surface = VolatilitySurfaceFromFile(
                file_path=file_path, risk_free_rate=risk_free_rate,
                option_type=option_type
            )
        except Exception as e:
            self.show_error(str(e))
            return

        self.thread = WorkerThread(vol_surface)
        self.thread.finished.connect(self.on_finished)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def on_finished(self, iv_values, ttm_grid, strike_grid):
        self.progress_bar.setFormat("Completed.")
        self.save_button.setEnabled(True)  # Enable the save button after successful plot generation
        try:
            self.canvas.plot_surface(ttm_grid, strike_grid, iv_values, self.thread.vol_surface._spot)
            logging.info("Volatility surface plotted successfully.")
        except Exception as e:
            self.show_error(f"Error plotting surface: {str(e)}")

    def on_save(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "", "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*)", options=options
        )
        if file_path:
            try:
                self.canvas.figure.savefig(file_path)
                QMessageBox.information(self, "Success", f"Plot saved to {file_path}")
                logging.info(f"Plot saved to {file_path}")
            except Exception as e:
                self.show_error(f"Failed to save plot: {str(e)}")

    def on_error(self, message):
        self.progress_bar.hide()
        self.show_error(message)

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)
```

**توضیحات کلاس `MainWindow`:**

1. **متد `__init__`:**
   - **تعریف عنوان و اندازه‌ی پنجره:** با استفاده از `setWindowTitle` و `setGeometry`, عنوان و اندازه‌ی پنجره را تنظیم می‌کنیم.
   - **ایجاد واسط کاربری:** با فراخوانی متد `init_ui`, بخش‌های مختلف واسط کاربری را ایجاد می‌کنیم.

2. **متد `init_ui`:**
   - **ایجاد لی‌آوت عمودی (`QVBoxLayout`):** برای ترتیب‌دهی بخش‌های مختلف به صورت عمودی.
   - **ایجاد گروه ورودی‌ها (`QGroupBox`):**
     - **فیلد انتخاب فایل:** یک `QLineEdit` برای نمایش مسیر فایل انتخاب شده و یک دکمه `Browse` برای باز کردن دیالوگ انتخاب فایل.
     - **فیلد وارد کردن نرخ بدون ریسک:** یک `QLineEdit` برای وارد کردن نرخ بدون ریسک به صورت درصدی.
     - **کادوی انتخاب نوع آپشن:** یک `QComboBox` برای انتخاب نوع آپشن (`Call` یا `Put`).
   - **ایجاد دکمه‌های Generate و Save:**
     - **دکمه Generate:** برای شروع فرآیند تولید سطح نوسان ضمنی.
     - **دکمه Save:** برای ذخیره نمودار تولید شده. این دکمه تا زمانی که نمودار تولید نشود غیرفعال است.
   - **ایجاد نوار پیشرفت (`QProgressBar`):** برای نمایش پیشرفت فرآیند.
   - **ایجاد بوم Matplotlib (`MatplotlibCanvas`):** برای نمایش نمودار سه‌بعدی سطح نوسان ضمنی.
   - **تنظیم لی‌آوت:** ترتیب‌دهی بخش‌ها به صورت عمودی و افزودن به پنجره اصلی.

3. **متد `on_browse`:**
   - **هدف:** باز کردن دیالوگ انتخاب فایل و تنظیم مسیر انتخاب شده در `QLineEdit`.
   - **روش:** با استفاده از `QFileDialog.getOpenFileName`, فایل مورد نظر را انتخاب کرده و مسیر آن را در فیلد ورودی نمایش می‌دهیم.

4. **متد `on_generate`:**
   - **هدف:** شروع فرآیند تولید سطح نوسان ضمنی.
   - **گام‌های انجام شده:**
     - **دریافت ورودی‌ها:** مسیر فایل، نرخ بدون ریسک و نوع آپشن را از فیلدهای ورودی دریافت می‌کنیم.
     - **بررسی صحت ورودی‌ها:** اطمینان حاصل می‌کنیم که مسیر فایل وارد شده باشد و نرخ بدون ریسک معتبر و غیرمنفی باشد.
     - **ایجاد نمونه از کلاس `VolatilitySurfaceFromFile`:** با استفاده از ورودی‌ها, شیء `vol_surface` ایجاد می‌کنیم.
     - **ایجاد و شروع `WorkerThread`:** با استفاده از شیء `vol_surface`, یک رشته‌ی کاری ایجاد کرده و آن را شروع می‌کنیم.
     - **اتصال سیگنال‌ها:** سیگنال‌های `finished` و `error` را به متدهای مربوطه متصل می‌کنیم تا در صورت اتمام یا بروز خطا, اقدامات لازم انجام شود.

5. **متد `on_finished`:**
   - **هدف:** دریافت نتایج محاسبات از `WorkerThread` و ترسیم نمودار سطح نوسان ضمنی.
   - **گام‌های انجام شده:**
     - **به‌روزرسانی نوار پیشرفت:** نوار پیشرفت را به وضعیت "Completed." تنظیم می‌کنیم.
     - **فعال‌سازی دکمه Save:** پس از موفقیت‌آمیز بودن ترسیم نمودار, دکمه Save را فعال می‌کنیم.
     - **ترسیم نمودار:** با استفاده از متد `plot_surface` از کلاس `MatplotlibCanvas`, نمودار سطح نوسان ضمنی را ترسیم می‌کنیم.
     - **ثبت لاگ‌ها:** لاگ‌های مربوط به موفقیت‌آمیز بودن ترسیم نمودار را ثبت می‌کنیم.

6. **متد `on_save`:**
   - **هدف:** ذخیره نمودار ترسیم شده به عنوان یک فایل تصویری.
   - **گام‌های انجام شده:**
     - **باز کردن دیالوگ ذخیره‌سازی:** با استفاده از `QFileDialog.getSaveFileName`, مکان و نام فایل ذخیره‌سازی را انتخاب می‌کنیم.
     - **ذخیره نمودار:** با استفاده از `self.canvas.figure.savefig`, نمودار را به مسیر انتخاب شده ذخیره می‌کنیم.
     - **نمایش پیام موفقیت یا خطا:** در صورت موفقیت, یک پیام اطلاعاتی نمایش داده می‌شود و در صورت بروز خطا, پیام خطا نمایش داده می‌شود.

7. **متد `on_error`:**
   - **هدف:** نمایش پیام خطا در صورت بروز مشکل در `WorkerThread`.
   - **گام‌های انجام شده:**
     - **پنهان‌سازی نوار پیشرفت:** نوار پیشرفت را مخفی می‌کنیم.
     - **نمایش پیام خطا:** با استفاده از `show_error`, پیام خطا را به کاربر نمایش می‌دهیم.

8. **متد `show_error`:**
   - **هدف:** نمایش پیام‌های خطا به کاربر به صورت پنجره‌ی پاپ‌آپ.
   - **روش:** با استفاده از `QMessageBox.critical`, پیام خطا را نمایش می‌دهیم.

---

## **۷. اجرای برنامه اصلی (Main Application)**

این بخش مسئول اجرای برنامه و نمایش پنجره‌ی اصلی واسط کاربری است.

```python
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
```

**توضیحات:**

1. **ایجاد برنامه‌ی `QApplication`:** با استفاده از `QApplication`, یک برنامه‌ی PyQt5 ایجاد می‌کنیم که تمامی اجزای GUI را مدیریت می‌کند.
2. **ایجاد و نمایش پنجره‌ی اصلی:** یک نمونه از کلاس `MainWindow` ایجاد کرده و آن را نمایش می‌دهیم.
3. **اجرای حلقه‌ی اصلی برنامه:** با استفاده از `app.exec_()`, حلقه‌ی اصلی برنامه را اجرا می‌کنیم تا پنجره‌ی GUI فعال بماند.
4. **خاتمه‌ی برنامه:** با استفاده از `sys.exit`, برنامه را پس از بسته شدن پنجره‌ی اصلی خاتمه می‌دهیم.

---

## **نتیجه‌گیری**

با اجرای این کد، شما ابزاری پیشرفته و جامع برای تولید و تحلیل سطح نوسان ضمنی در اختیار دارید که از روش‌های علمی و ریاضیاتی بهره‌برداری می‌کند. استفاده از **PyQt5** برای ایجاد واسط کاربری گرافیکی، **Matplotlib** برای ترسیم نمودارهای سه‌بعدی، و **Scipy** و **Numpy** برای محاسبات ریاضی، نشان‌دهنده‌ی توانمندی شما در ترکیب ابزارهای مختلف به‌منظور حل مسائل پیچیده مالی است. دوست دار شما - نوید رمضانی

**ویژگی‌های برجسته این ابزار شامل:**

1. **بارگذاری دقیق داده‌ها:** با استفاده از کلاس‌های `Option` و `VolatilitySurfaceFromFile`, داده‌های آپشن‌ها به‌صورت دقیق بارگذاری و پردازش می‌شوند.
2. **محاسبه نوسان ضمنی:** با بهره‌گیری از مدل Black-Scholes و روش Brent's method, نوسان ضمنی با دقت بالا استخراج می‌شود.
3. **ترسیم نمودار تعاملی:** با استفاده از `MatplotlibCanvas`, نمودار سه‌بعدی سطح نوسان ضمنی به‌صورت تعاملی و زیبا ترسیم شده و قابلیت تعامل با کاربر فراهم شده است.
4. **مدیریت کارآمد فرآیند:** با بهره‌گیری از `WorkerThread`, محاسبات پیچیده به‌صورت غیرهمزمان اجرا می‌شوند تا واسط کاربری اصلی مسدود نشود و تجربه کاربری بهتری ارائه گردد.
5. **ثبت دقیق لاگ‌ها:** با استفاده از کتابخانه‌ی `logging`, تمامی مراحل و خطاها به‌صورت دقیق ثبت و قابلیت پیگیری و رفع مشکلات فراهم شده است.
6. **امکان ذخیره نمودار:** کاربر می‌تواند نمودار تولید شده را به‌صورت فایل‌های تصویری ذخیره کند تا در آینده بتواند به‌راحتی از آن استفاده نماید.

