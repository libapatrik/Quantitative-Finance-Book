import threading
import time
import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import implied_vol

# Use dark background style for a professional terminal aesthetic
plt.style.use('dark_background')

# --- IBKR Application Class ---
# Inherits from EWrapper (to receive data) and EClient (to send requests)
class LiveSurfaceApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.id_map = {}           # reqId -> (exp, strike)
        self.expirations = []
        self.strikes = []
        self.spot_price = 0
        self.underlying_conId = 0
        self.resolved = threading.Event()
        self.chain_resolved = threading.Event()
        # option price storage for IV calc
        self.option_prices = {}    # reqId -> {'bid': x, 'ask': y}
        self.option_meta = {}      # reqId -> {'strike': K, 'T': T, 'opt_type': 'call'/'put'}

        # trading state
        self.atm_iv_history = []      # rolling ATM IV samples
        self.current_position = None  # 'long_straddle', 'short_straddle', or None
        self.order_id = 0             # IBKR order ID counter
        self.pending_orders = {}      # orderId -> order details
        self.filled_orders = {}       # orderId -> fill details

    # Callback triggered when connection to TWS/Gateway is successful
    def connectAck(self):
        print("TWS Acknowledged Connection")

    def error(self, reqId, errorCode, errorString, advancedOrderRejectionJson=""):
        # filter out connectivity and delayed data notifications
        if errorCode not in [2104, 2106, 2158, 10091, 10167]:
            print(f"IBKR Msg {reqId}: {errorCode} - {errorString}")

    # Callback receiving contract details like the unique contract ID (conId)
    def contractDetails(self, reqId, contractDetails):
        self.underlying_conId = contractDetails.contract.conId
        self.resolved.set()

    def tickPrice(self, reqId, tickType, price, attrib):
        # underlying spot: tickType 4/9 = real-time Last/Close, 68/75 = delayed
        if reqId == 999 and tickType in [4, 9, 68, 75] and price > 0:
            self.spot_price = price
        # option bid/ask (reqId >= 1000)
        elif reqId >= 1000 and price > 0:
            if reqId not in self.option_prices:
                self.option_prices[reqId] = {}
            if tickType in [1, 66]:    # bid (real-time=1, delayed=66)
                self.option_prices[reqId]['bid'] = price
            elif tickType in [2, 67]:  # ask (real-time=2, delayed=67)
                self.option_prices[reqId]['ask'] = price

    # Callback receiving the list of strikes and expirations for the asset
    def securityDefinitionOptionParameter(self, reqId, exchange, underlyingConId, tradingClass, multiplier, expirations, strikes):
        if exchange == "SMART":
            self.expirations = sorted(list(expirations))
            self.strikes = sorted(list(strikes))
            self.chain_resolved.set()

    def tickOptionComputation(self, reqId, tickType, tickAttrib, impliedVol, delta, optPrice, pvDividend, gamma, vega, theta, undPrice):
        # not used with delayed data, but keep for real-time fallback
        pass

    # --- Trading callbacks ---
    def nextValidId(self, orderId):
        self.order_id = orderId
        print(f"Next valid order ID: {orderId}")

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice,
                    permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        print(f"Order {orderId}: {status} | filled={filled} remaining={remaining} avgPrice={avgFillPrice}")
        if orderId in self.pending_orders:
            self.pending_orders[orderId]['status'] = status
            self.pending_orders[orderId]['filled'] = filled
            if status == 'Filled':
                self.filled_orders[orderId] = self.pending_orders.pop(orderId)
                self.filled_orders[orderId]['avgFillPrice'] = avgFillPrice

    def execDetails(self, reqId, contract, execution):
        print(f"Exec: {execution.side} {execution.shares} {contract.symbol} {contract.right} "
              f"@ {execution.price} (orderId={execution.orderId})")

# Wrapper function to run the client messaging loop
def run_loop(app):
    app.run()


# --- Signal Generation ---
def compute_atm_iv(app, r=0.05):
    """Interpolate IV at the ATM strike from current surface data."""
    spot = app.spot_price
    if spot == 0:
        return None

    # collect all valid IVs with strikes
    ivs_by_strike = {}
    for rid, prices in app.option_prices.items():
        if 'bid' not in prices or 'ask' not in prices:
            continue
        mid = (prices['bid'] + prices['ask']) / 2
        meta = app.option_meta.get(rid)
        if meta is None:
            continue
        try:
            iv = implied_vol(mid, spot, meta['strike'], r, meta['T'], meta['opt_type'])
            if 0.01 < iv < 2.0:
                strike = meta['strike']
                # keep shortest expiry only for ATM
                if strike not in ivs_by_strike:
                    ivs_by_strike[strike] = iv
        except:
            pass

    if len(ivs_by_strike) < 2:
        return None

    # find two strikes nearest to spot
    strikes = sorted(ivs_by_strike.keys())
    below = [s for s in strikes if s <= spot]
    above = [s for s in strikes if s > spot]

    if not below or not above:
        # all on one side, just take nearest
        nearest = min(strikes, key=lambda s: abs(s - spot))
        return ivs_by_strike[nearest]

    k_lo, k_hi = below[-1], above[0]
    iv_lo, iv_hi = ivs_by_strike[k_lo], ivs_by_strike[k_hi]

    # linear interpolation
    w = (spot - k_lo) / (k_hi - k_lo)
    atm_iv = iv_lo * (1 - w) + iv_hi * w
    return atm_iv


def generate_signal(app, lookback=20, threshold=0.03):
    """
    Returns: 'sell', 'buy', or None
    - 'sell' if current ATM IV > avg + threshold (vol rich)
    - 'buy' if current ATM IV < avg - threshold (vol cheap)
    """
    current_iv = compute_atm_iv(app)
    if current_iv is None:
        return None

    app.atm_iv_history.append(current_iv)
    if len(app.atm_iv_history) > lookback:
        app.atm_iv_history.pop(0)

    if len(app.atm_iv_history) < lookback:
        print(f"ATM IV: {current_iv:.4f} | warming up ({len(app.atm_iv_history)}/{lookback})")
        return None

    avg_iv = np.mean(app.atm_iv_history)
    deviation = current_iv - avg_iv
    print(f"ATM IV: {current_iv:.4f} | avg: {avg_iv:.4f} | dev: {deviation:+.4f}")

    if current_iv > avg_iv + threshold:
        return 'sell'
    elif current_iv < avg_iv - threshold:
        return 'buy'
    return None


# --- Order Execution ---
def create_straddle_order(app, action, quantity=1):
    """
    Place ATM straddle order (call + put at same strike).
    action: 'BUY' or 'SELL'
    """
    spot = app.spot_price
    if spot == 0:
        print("No spot price available")
        return

    atm_strike = round(spot)  # nearest $1 strike

    # use second expiry to avoid 0-DTE issues
    today = time.strftime("%Y%m%d")
    valid_exps = [e for e in app.expirations if e > today]
    if len(valid_exps) < 2:
        print("Not enough expirations available")
        return
    front_exp = valid_exps[1]  # skip 0-DTE

    # ATM Call leg
    call_contract = Contract()
    call_contract.symbol = "SPY"
    call_contract.secType = "OPT"
    call_contract.exchange = "SMART"
    call_contract.currency = "USD"
    call_contract.lastTradeDateOrContractMonth = front_exp
    call_contract.strike = atm_strike
    call_contract.right = "C"

    call_order = Order()
    call_order.action = action
    call_order.totalQuantity = quantity
    call_order.orderType = "MKT"

    print(f"Placing {action} CALL @ {atm_strike} exp {front_exp}")
    app.pending_orders[app.order_id] = {
        'action': action, 'strike': atm_strike, 'right': 'C', 'exp': front_exp, 'status': 'Submitted'
    }
    app.placeOrder(app.order_id, call_contract, call_order)
    app.order_id += 1

    # ATM Put leg
    put_contract = Contract()
    put_contract.symbol = "SPY"
    put_contract.secType = "OPT"
    put_contract.exchange = "SMART"
    put_contract.currency = "USD"
    put_contract.lastTradeDateOrContractMonth = front_exp
    put_contract.strike = atm_strike
    put_contract.right = "P"

    put_order = Order()
    put_order.action = action
    put_order.totalQuantity = quantity
    put_order.orderType = "MKT"

    print(f"Placing {action} PUT @ {atm_strike} exp {front_exp}")
    app.pending_orders[app.order_id] = {
        'action': action, 'strike': atm_strike, 'right': 'P', 'exp': front_exp, 'status': 'Submitted'
    }
    app.placeOrder(app.order_id, put_contract, put_order)
    app.order_id += 1


# --- Trading Loop ---
def trading_loop(app, check_interval=30, lookback=20, threshold=0.03):
    """
    Main trading loop - runs alongside visualization.
    """
    print(f" --- TRADING LOOP STARTED (interval={check_interval}s, lookback={lookback}, threshold={threshold}) ---")
    while True:
        time.sleep(check_interval)

        signal = generate_signal(app, lookback=lookback, threshold=threshold)

        if signal == 'sell' and app.current_position != 'short_straddle':
            if app.current_position == 'long_straddle':
                # close existing long first
                print("Closing existing LONG straddle...")
                create_straddle_order(app, 'SELL', quantity=1)
            print(f"SIGNAL: SELL STRADDLE - ATM IV elevated")
            create_straddle_order(app, 'SELL', quantity=1)
            app.current_position = 'short_straddle'

        elif signal == 'buy' and app.current_position != 'long_straddle':
            if app.current_position == 'short_straddle':
                # close existing short first
                print("Closing existing SHORT straddle...")
                create_straddle_order(app, 'BUY', quantity=1)
            print(f"SIGNAL: BUY STRADDLE - ATM IV depressed")
            create_straddle_order(app, 'BUY', quantity=1)
            app.current_position = 'long_straddle'


def start_app(symbol="SPY"):
    app = LiveSurfaceApp()
    app.connect("127.0.0.1", 7497, clientId=1)
    api_thread = threading.Thread(target=run_loop, args=(app,), daemon=True)
    api_thread.start()
    time.sleep(5)

    if app.isConnected():
        print("Connected successfully!")
        app.reqMarketDataType(3)  # use delayed data 
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        app.reqContractDetails(1, contract) # requesting the contract ID
        app.resolved.wait(timeout=5)

        app.reqMktData(999, contract, "", False, False, [])
        while app.spot_price == 0: time.sleep(0.1)
        spot = app.spot_price
        print(f"Spot price: {spot}")

        app.reqSecDefOptParams(2, symbol, "", "STK", app.underlying_conId)
        app.chain_resolved.wait(timeout=5)
        
        today = time.strftime("%Y%m%d")
        target_exps = [e for e in app.expirations if e >= today][:6]
        target_strikes = [s for s in app.strikes if spot * 0.98 <= s <= spot * 1.02]  # ±2% ATM only

        # live data feed requests for option prices
        from datetime import datetime
        today_dt = datetime.strptime(today, "%Y%m%d")

        req_id = 1000
        for exp in target_exps:
            exp_dt = datetime.strptime(exp, "%Y%m%d")
            T = (exp_dt - today_dt).days / 365.0
            T = max(T, 1/365)  # floor at 1 day

            for strike in target_strikes:
                opt = Contract()
                opt.symbol = symbol
                opt.secType = "OPT"
                opt.exchange = "SMART"
                opt.currency = "USD"
                opt.lastTradeDateOrContractMonth = exp
                opt.strike = strike
                # use OTM options: puts below spot, calls above
                opt_type = "C" if strike >= spot else "P"
                opt.right = opt_type

                app.id_map[req_id] = (exp, strike)
                app.option_meta[req_id] = {
                    'strike': strike,
                    'T': T,
                    'opt_type': 'call' if opt_type == 'C' else 'put'
                }

                app.reqMktData(req_id, opt, "", False, False, [])
                req_id += 1
                time.sleep(0.01)

    return app

# Visualizer
class PlotState:

    def __init__(self):
        self.is_locked = False 

    def toggle(self, event): 
        self.is_locked = not self.is_locked  # flip it
        btn_label.set_text("UNLOCK UPDATES" if self.is_locked else "LOCK UPDATES")
        plt.draw()

def live_desktop_plot(app):
    plt.ion()
    fig = plt.figure(figsize=(16, 9))
    fig.canvas.manager.set_window_title("Live Surface Plot")
    
    ax_3d = plt.subplot2grid((1, 3), (0, 0), colspan=2, projection='3d')
    ax_skew = plt.subplot2grid((1, 3), (0, 2))

    state = PlotState()
    ax_button = plt.axes([0.42, 0.03, 0.12, 0.04])
    global btn_label
    btn = Button(ax_button, "LOCK UPDATES") # button blocks the updates
    btn_label = btn.label
    
    btn.on_clicked(state.toggle) # to toggle the state
 
    print(" --- STARTING LIVE VOL SURFACE PLOT --- ")

    r = 0.05  # risk-free rate assumption
    try:
        while True:
            if not state.is_locked:
                current_data = []
                spot = app.spot_price
                for rid, prices in app.option_prices.items():
                    if 'bid' not in prices or 'ask' not in prices:
                        continue
                    mid = (prices['bid'] + prices['ask']) / 2
                    meta = app.option_meta.get(rid)
                    if meta is None:
                        continue
                    try:
                        iv = implied_vol(mid, spot, meta['strike'], r, meta['T'], meta['opt_type'])
                        if 0.01 < iv < 2.0:  # sanity filter
                            exp, strike = app.id_map[rid]
                            current_data.append({'Expiry': exp, 'Strike': strike, 'IV': iv})
                    except:
                        pass  # solver failed, skip
                if len(current_data) > 10: # do visualize otherwise not enough data
                    df = pd.DataFrame(current_data)
                    pivot = df.pivot_table(index='Expiry', columns='Strike', values='IV').sort_index().sort_index(axis=1) # table of ivs
                    # interpolation back/forward fill the values
                    pivot = pivot.interpolate(method='linear', axis=0).bfill().ffill()

                    X, Y_idx = np.meshgrid(pivot.columns, np.arange(len(pivot.index)))
                    Z = pivot.values

                    curr_elev, curr_azim = ax_3d.elev, ax_3d.azim # for plot redrawing
                    ax_3d.clear()
                    ax_3d.plot_surface(X, Y_idx, Z, cmap='viridis', alpha=0.8)
                    ax_3d.set_yticks(np.arange(len(pivot.index)))
                    ax_3d.set_yticklabels(pivot.index)
                    ax_3d.set_title(f'Live Surface Plot | {time.strftime("%H-%M-%S")}')
                    ax_3d.view_init(elev=curr_elev, azim=curr_azim) # reset the camera

                    ax_skew.clear()
                    nearest_exp = pivot.index[0]
                    skew_data = pivot.iloc[0]
                    ax_skew.set_title(f'FRONT-MONTH SKEW | {nearest_exp} | {time.strftime("%H-%M-%S")}')
                    ax_skew.axvline(x=app.spot_price, color='r', ls='--', lw=2)
                    ax_skew.plot(skew_data.index, skew_data.values, 'co-', lw=2, markersize=5)

            plt.pause(0.5) # pause between repaint

    except KeyboardInterrupt:
        plt.close()
        app.disconnect()
        return


if __name__ == "__main__":
    app_instance = start_app()
    print(" --- STARTING LIVE DESKTOP PLOT --- ")
    time.sleep(10)

    # start trading loop in background thread
    trading_thread = threading.Thread(
        target=trading_loop,
        args=(app_instance,),
        kwargs={'check_interval': 30, 'lookback': 20, 'threshold': 0.03},
        daemon=True
    )
    trading_thread.start()

    live_desktop_plot(app_instance)


    

