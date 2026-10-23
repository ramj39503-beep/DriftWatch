"""
DriftWatch — AI Model Health Monitor
Run: streamlit run app.py
Note: bundled/synthetic data is clearly labeled demo data. Replace it with approved real data.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="DriftWatch | AI Model Health", page_icon="〽️", layout="wide")
st.markdown("""
<style>
.stApp {background: radial-gradient(ellipse at 12% 0%, #172b54 0%, #0a1327 42%, #070d19 100%); color:#edf3ff}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#101d37,#080e1c); border-right:1px solid #263b63}
h1,h2,h3 {color:#f1f5ff}
.hero {padding:25px 28px;border:1px solid #2c4777;border-radius:20px;background:linear-gradient(110deg,rgba(27,54,101,.92),rgba(12,24,47,.86)), radial-gradient(circle at 85% 20%,#7047ff,transparent 35%);box-shadow:0 14px 45px #0005}
.hero h1 {font-size:2.65rem; margin:0; background:linear-gradient(90deg,#fff,#9bc4ff,#c4a6ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero p {color:#c8d8f5;font-size:1.05rem}
.metric-card {background:linear-gradient(135deg,#142847,#0d192d);border:1px solid #2a416a;border-radius:15px;padding:15px}
.small-muted {color:#9eb2d5;font-size:.88rem}
div[data-testid="stMetric"] {background:linear-gradient(135deg,#142847,#0c172b);border:1px solid #2a416a;padding:15px;border-radius:14px}
.stButton>button {border-radius:10px;border:1px solid #7a75ff;background:linear-gradient(90deg,#5746e8,#8157f4);color:white}
</style>
""", unsafe_allow_html=True)

FEATURES = ["age", "income", "balance", "transactions", "credit_score"]
@st.cache_data
def make_demo(seed=42):
    rng=np.random.default_rng(seed)
    X,y=make_classification(n_samples=1600,n_features=5,n_informative=4,n_redundant=0,
                            n_clusters_per_class=2,weights=[.68,.32],flip_y=.025,random_state=seed)
    X=pd.DataFrame(X,columns=FEATURES)
    # Transform standard synthetic values into human-readable ranges.
    X["age"]=np.clip(38+X["age"]*12,18,85).round(0)
    X["income"]=np.clip(42000+X["income"]*16000,8000,180000).round(0)
    X["balance"]=np.clip(9000+X["balance"]*6500,0,65000).round(0)
    X["transactions"]=np.clip(12+X["transactions"]*5,0,50).round(0)
    X["credit_score"]=np.clip(680+X["credit_score"]*55,300,850).round(0)
    ref=X.copy(); ref["target"]=y
    # Deliberately shifted demo batch to show drift detection working.
    cur=X.sample(500,random_state=seed+1).copy()
    cur["income"]=np.clip(cur["income"]*1.32+9000,8000,220000)
    cur["transactions"]=np.clip(cur["transactions"]+rng.normal(4,3,len(cur)),0,55)
    cur["target"]=ref.loc[cur.index,"target"].to_numpy()
    return ref.reset_index(drop=True),cur.reset_index(drop=True)

def psi(expected, actual, bins=10):
    expected=np.asarray(expected,dtype=float); actual=np.asarray(actual,dtype=float)
    cuts=np.unique(np.quantile(expected,np.linspace(0,1,bins+1)))
    if len(cuts)<3: return 0.0
    cuts[0],cuts[-1]=-np.inf,np.inf
    e=np.histogram(expected,bins=cuts)[0]/len(expected)
    a=np.histogram(actual,bins=cuts)[0]/len(actual)
    e=np.clip(e,1e-5,None); a=np.clip(a,1e-5,None)
    return float(np.sum((a-e)*np.log(a/e)))

def drift_table(reference,current):
    rows=[]
    for f in FEATURES:
        a=reference[f].dropna(); b=current[f].dropna()
        ks=float(ks_2samp(a,b).statistic) if len(a) and len(b) else 0
        p=float(ks_2samp(a,b).pvalue) if len(a) and len(b) else 1
        pval=psi(a,b)
        rows.append({"Feature":f.replace("_"," ").title(),"PSI":pval,"KS statistic":ks,"KS p-value":p,
                     "Status":"Drift detected" if (pval>=0.20 or p<0.05) else "Stable"})
    return pd.DataFrame(rows)

st.sidebar.markdown("## 〽️ DriftWatch")
st.sidebar.caption("AI Model Health Monitor")
page=st.sidebar.radio("Navigation",["Overview","Data Drift","Model Performance","How it works"])
st.sidebar.divider()
st.sidebar.markdown("**System status**")
st.sidebar.success("Monitoring ready")
st.sidebar.caption("Demo mode • Synthetic data")
reference, current=make_demo()
with st.sidebar.expander("Use your own CSV"):
    uploaded_ref=st.file_uploader("Reference CSV",type=["csv"],key="ref")
    uploaded_cur=st.file_uploader("Current batch CSV",type=["csv"],key="cur")
    st.caption("CSV must include age, income, balance, transactions, credit_score. Optional target column: 0/1.")
    if uploaded_ref is not None and uploaded_cur is not None:
        try:
            r=pd.read_csv(uploaded_ref); c=pd.read_csv(uploaded_cur)
            missing=[x for x in FEATURES if x not in r.columns or x not in c.columns]
            if missing: st.error("Missing required columns: "+", ".join(missing))
            else:
                reference=r.copy(); current=c.copy()
                if "target" not in reference: reference["target"]=0
                if "target" not in current: current["target"]=0
                st.success("Uploaded datasets loaded for this session.")
        except Exception as e: st.error(f"Could not read CSV: {e}")

# Train a reproducible baseline model on reference dataset.
X=reference[FEATURES].apply(pd.to_numeric,errors="coerce")
y=pd.to_numeric(reference["target"],errors="coerce").fillna(0).astype(int).clip(0,1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=42,stratify=y if y.nunique()>1 else None)
model=Pipeline([("imputer",SimpleImputer(strategy="median")),("forest",RandomForestClassifier(n_estimators=120,random_state=42,class_weight="balanced"))])
model.fit(X_train,y_train)
pred=model.predict(X_test)
metrics={"Accuracy":accuracy_score(y_test,pred),"Precision":precision_score(y_test,pred,zero_division=0),
         "Recall":recall_score(y_test,pred,zero_division=0),"F1 score":f1_score(y_test,pred,zero_division=0)}
drift=drift_table(reference,current)
n_drift=int((drift["Status"]=="Drift detected").sum())
risk="Attention required" if n_drift else "Healthy"
if page=="Overview":
    st.markdown('<div class="hero"><h1>DriftWatch</h1><p>Detect. Analyze. Stay ahead.</p><p>Monitor changes in real-world data and model quality before they undermine prediction reliability.</p></div>',unsafe_allow_html=True)
    st.write("")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Reference samples",f"{len(reference):,}")
    c2.metric("Current batch",f"{len(current):,}")
    c3.metric("Features with drift",f"{n_drift}/{len(FEATURES)}",delta="Review recommended" if n_drift else "Stable")
    c4.metric("Model accuracy",f"{metrics['Accuracy']:.1%}")
    left,right=st.columns([1.3,1])
    with left:
        st.subheader("Feature drift overview")
        fig,ax=plt.subplots(figsize=(8,3.6)); fig.patch.set_facecolor("#101b30"); ax.set_facecolor("#101b30")
        vals=drift["PSI"].values
        ax.bar(drift["Feature"],vals,color=["#a78bfa" if v>=.2 else "#38bdf8" for v in vals])
        ax.axhline(.1,color="#fbbf24",linestyle="--",label="PSI 0.10 watch")
        ax.axhline(.2,color="#fb7185",linestyle="--",label="PSI 0.20 alert")
        ax.tick_params(axis="x",rotation=18,colors="#dce7ff"); ax.tick_params(axis="y",colors="#dce7ff")
        ax.set_ylabel("Population Stability Index",color="#dce7ff"); ax.legend(facecolor="#172744",labelcolor="white",fontsize=8)
        for sp in ax.spines.values(): sp.set_color("#304666")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with right:
        st.subheader("Model performance")
        for name,value in metrics.items(): st.progress(float(value),text=f"{name}: {value:.3f}")
        st.info("A strong score on a held-out reference split does not guarantee future performance. Monitor labelled outcomes over time.")
    st.subheader("Latest monitoring findings")
    st.dataframe(drift[["Feature","PSI","KS statistic","KS p-value","Status"]].style.format({"PSI":"{:.3f}","KS statistic":"{:.3f}","KS p-value":"{:.4f}"}),use_container_width=True,hide_index=True)
    st.caption("All results above use synthetic demonstration data unless you upload your own CSV.")
elif page=="Data Drift":
    st.title("Data Drift Analysis")
    st.write("Compare the reference feature distributions with the current batch. PSI and the two-sample Kolmogorov–Smirnov test are screening signals, not proof that a model is wrong.")
    st.dataframe(drift.style.format({"PSI":"{:.4f}","KS statistic":"{:.4f}","KS p-value":"{:.5f}"}),use_container_width=True,hide_index=True)
    feature=st.selectbox("Inspect feature",FEATURES)
    fig,ax=plt.subplots(figsize=(9,3.5)); fig.patch.set_facecolor("#101b30"); ax.set_facecolor("#101b30")
    ax.hist(reference[feature].dropna(),bins=24,density=True,alpha=.55,label="Reference",color="#38bdf8")
    ax.hist(current[feature].dropna(),bins=24,density=True,alpha=.55,label="Current batch",color="#c084fc")
    ax.set_title(f"{feature.replace('_',' ').title()} distribution",color="white")
    ax.tick_params(colors="#dce7ff"); ax.legend(facecolor="#172744",labelcolor="white")
    for sp in ax.spines.values(): sp.set_color("#304666")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.download_button("Download drift report CSV",drift.to_csv(index=False).encode(),file_name="driftwatch_drift_report.csv",mime="text/csv")
    st.warning("Thresholds are configurable heuristics. Validate thresholds against your use case and false-alarm costs.")
elif page=="Model Performance":
    st.title("Model Performance")
    st.write("A Random Forest baseline is trained on the reference data and evaluated on a held-out split.")
    cols=st.columns(4)
    for col,(name,value) in zip(cols,metrics.items()): col.metric(name,f"{value:.3f}")
    cm=confusion_matrix(y_test,pred,labels=[0,1])
    fig,ax=plt.subplots(figsize=(5,3.8)); fig.patch.set_facecolor("#101b30"); ax.set_facecolor("#101b30")
    im=ax.imshow(cm,cmap="Purples")
    ax.set_xticks([0,1],["Predicted 0","Predicted 1"]); ax.set_yticks([0,1],["Actual 0","Actual 1"])
    ax.tick_params(colors="#dce7ff")
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha="center",va="center",color="white",fontsize=13)
    ax.set_title("Confusion matrix",color="white"); fig.colorbar(im,ax=ax); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption("Performance metrics are computed only where target labels exist. For a real deployment, monitor delayed labels and compare against a defined baseline.")
else:
    st.title("How DriftWatch works")
    st.markdown("""
1. **Load data:** use the built-in synthetic demo or upload reference and current CSV files.
2. **Train baseline:** a scikit-learn Random Forest is trained using a held-out evaluation split.
3. **Measure drift:** compare each numeric feature with PSI and the two-sample KS test.
4. **Inspect impact:** review model metrics and drift results in charts and tables.
5. **Take action:** investigate data pipelines, segment changes, and evaluate retraining only after validation.
    """)
    st.subheader("What the signals mean")
    st.markdown("- **PSI:** rough measure of distribution change; commonly used alert bands are heuristics.\n- **KS test:** tests whether two numeric samples may come from different distributions.\n- **Model metrics:** accuracy, precision, recall and F1 on a held-out labelled split.\n- **Evidently:** can be added for richer production monitoring reports; this starter app uses transparent PSI/KS calculations so it runs consistently across versions.")
    st.warning("This is a student project prototype, not an automatic production alerting service. It does not monitor continuously or connect to a live model endpoint.")
st.divider()
st.caption("DriftWatch • Student project prototype • Synthetic data is clearly labeled • Validate before real-world use")
