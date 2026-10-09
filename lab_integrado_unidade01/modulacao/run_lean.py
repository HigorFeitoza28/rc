import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUT=Path(r"C:\Users\UsoPessoal\Desktop\projects\REDES\lab2-real\rc\lab_integrado_unidade01\modulacao")

SR=44100; BD=0.5; FL=440; FH=880; TH=660
def gen(f,d,sr=SR):
    n=int(sr*d); t=np.linspace(0,d,n,endpoint=False)
    return np.sin(2*np.pi*f*t)*np.hanning(n)
def det(seg,sr=SR):
    fft=np.fft.fft(seg); freqs=np.fft.fftfreq(len(fft),1/sr)
    mag=np.abs(fft[:len(fft)//2]); fp=freqs[:len(freqs)//2]
    return abs(fp[np.argmax(mag)])
def b2b(f,thr=TH): return '1' if f>thr else '0'
def noise(sig,snr_db):
    sp=np.mean(sig**2); snr=10**(snr_db/10); pw=sp/snr
    return sig+np.random.normal(0,np.sqrt(pw),len(sig))
def enc_nrz(bits,bd=BD,fl=FL,fh=FH,sr=SR): return np.concatenate([gen(fh if b=='1' else fl,bd,sr) for b in bits])
def dec_nrz(sig,n,bd=BD,fl=FL,fh=FH,sr=SR):
    thr=(fl+fh)/2; spb=int(sr*bd); o=[]
    for i in range(n):
        s,e=i*spb,(i+1)*spb
        if e>len(sig): break
        o.append(b2b(det(sig[s+spb//4:e-spb//4],sr),thr))
    return ''.join(o)
def enc_man(bits,bd=BD,fl=FL,fh=FH,sr=SR):
    half=bd/2; segs=[]
    for b in bits:
        if b=='1': segs+=[gen(fh,half,sr),gen(fl,half,sr)]
        else: segs+=[gen(fl,half,sr),gen(fh,half,sr)]
    return np.concatenate(segs)
def dec_man(sig,n,bd=BD,fl=FL,fh=FH,sr=SR):
    thr=(fl+fh)/2; spb=int(sr*bd); o=[]
    for i in range(n):
        s,e=i*spb,(i+1)*spb
        if e>len(sig): break
        mid=s+spb//2; pad=spb//8
        b1=b2b(det(sig[s+pad:mid-pad],sr),thr); b2=b2b(det(sig[mid+pad:e-pad],sr),thr)
        o.append('1' if b1=='1' and b2=='0' else '0' if b1=='0' and b2=='1' else '?')
    return ''.join(o)
def enc_nrzi(bits,bd=BD,fl=FL,fh=FH,sr=SR):
    cur=fl; segs=[]
    for b in bits:
        if b=='1': cur=fh if cur==fl else fl
        segs.append(gen(cur,bd,sr))
    return np.concatenate(segs)
def dec_nrzi(sig,n,bd=BD,fl=FL,fh=FH,sr=SR):
    thr=(fl+fh)/2; spb=int(sr*bd); o=[]; prev='0'
    for i in range(n):
        s,e=i*spb,(i+1)*spb
        if e>len(sig): break
        cur=b2b(det(sig[s+spb//4:e-spb//4],sr),thr)
        o.append('1' if cur!=prev else '0'); prev=cur
    return ''.join(o)
def ber(o,d): return sum(1 for a,b in zip(o,d) if a!=b)/len(o) + abs(len(o)-len(d))/len(o) if o else 0
# actually ber formula above double counts; fix:
def ber2(o,d):
    if not o: return 0
    return (sum(1 for a,b in zip(o,d) if a!=b) + abs(len(o)-len(d)))/len(o)

def experimento(bits,snrs,bd=BD,fl=FL,fh=FH,sr=SR,trials=5):
    res={'NRZ':[],'Manchester':[],'NRZI':[]}; n=len(bits)
    for snr in snrs:
        b={'NRZ':[],'Manchester':[],'NRZI':[]}
        for _ in range(trials):
            b['NRZ'].append(ber2(bits, dec_nrz(noise(enc_nrz(bits,bd,fl,fh,sr),snr),n,bd,fl,fh,sr)))
            b['Manchester'].append(ber2(bits, dec_man(noise(enc_man(bits,bd,fl,fh,sr),snr),n,bd,fl,fh,sr)))
            b['NRZI'].append(ber2(bits, dec_nrzi(noise(enc_nrzi(bits,bd,fl,fh,sr),snr),n,bd,fl,fh,sr)))
        for k in res: res[k].append(float(np.mean(b[k])))
        print(f"  snr {snr:>4} dB -> NRZ {res['NRZ'][-1]:.3f} Man {res['Manchester'][-1]:.3f} NRZI {res['NRZI'][-1]:.3f}")
    return res

def plot_res(res,snrs,titulo,fn):
    plt.figure(figsize=(10,5))
    cols={'NRZ':'#1f77b4','Manchester':'#ff7f0e','NRZI':'#2ca02c'}
    for k,v in res.items(): plt.plot(snrs,v,marker='o',ms=4,lw=1.6,label=k,color=cols[k])
    plt.xlabel('SNR (dB) — canal piora para a direita'); plt.ylabel('BER'); plt.title(titulo,fontsize=11)
    plt.ylim(-0.02,0.62); plt.grid(True,alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(OUT/fn,dpi=150); plt.close(); print(f"saved {fn}")

np.random.seed(7)
bits_long=''.join(np.random.choice(list('01'),120))
print(f"bits_long={len(bits_long)}")
# SNR mais largo mas step 5 para ser rapido; depois o notebook pode usar step 2 se quiser
snrs=np.arange(30,-21,-5)  # 30,25,..., -20  -> 11 niveis
print("SNRs:",snrs.tolist())

print("\n=== LEAN A padrao ===")
resA=experimento(bits_long,snrs,trials=5)
plot_res(resA,snrs,"Q1.1 LEAN — A padrao (0.5s | 440/880 Hz) | SNR x BER","q11_LEAN_A.png")
import csv
with open(OUT/"q11_LEAN_A.csv",'w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['SNR_dB','NRZ','Manchester','NRZI'])
    for i,s in enumerate(snrs): w.writerow([s,resA['NRZ'][i],resA['Manchester'][i],resA['NRZI'][i]])

print("\n=== LEAN B1 0.1s 440/550 ===")
resB1=experimento(bits_long,snrs,bd=0.1,fl=440,fh=550,trials=5)
plot_res(resB1,snrs,"Q1.1 LEAN — B1 (0.1s | 440/550 Hz) | SNR x BER","q11_LEAN_B1.png")
with open(OUT/"q11_LEAN_B1.csv",'w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['SNR_dB','NRZ','Manchester','NRZI'])
    for i,s in enumerate(snrs): w.writerow([s,resB1['NRZ'][i],resB1['Manchester'][i],resB1['NRZI'][i]])

print("\n=== LEAN B2 0.5s 300/1200 (afastado, rapido de computar) ===")
resB2=experimento(bits_long,snrs,bd=0.5,fl=300,fh=1200,trials=5)
plot_res(resB2,snrs,"Q1.1 LEAN — B2 (0.5s | 300/1200 Hz) | SNR x BER","q11_LEAN_B2.png")
with open(OUT/"q11_LEAN_B2.csv",'w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['SNR_dB','NRZ','Manchester','NRZI'])
    for i,s in enumerate(snrs): w.writerow([s,resB2['NRZ'][i],resB2['Manchester'][i],resB2['NRZI'][i]])

def pts(res,snrs,label):
    print(f"\n-- {label} --")
    for k,v in res.items():
        pe=next((snrs[i] for i,x in enumerate(v) if x>0),None)
        ft=next((snrs[i] for i,x in enumerate(v) if x>=0.5),None)
        print(f"{k:<12} 1o erro {pe} dB  falha {ft} dB")

pts(resA,snrs,"LEAN A"); pts(resB1,snrs,"LEAN B1"); pts(resB2,snrs,"LEAN B2")

def tabela(res,snrs,titulo):
    ls=[f"### {titulo}","", "| Modulacao | 1o erro (BER>0) | Falha total (BER>=0.5) | Margem |","|:---|:---:|:---:|:---:|"]
    for k,v in res.items():
        pe=next((snrs[i] for i,x in enumerate(v) if x>0),None)
        ft=next((snrs[i] for i,x in enumerate(v) if x>=0.5),None)
        pe_s=f"{pe} dB" if pe is not None else "nao atingido"
        ft_s=f"{ft} dB" if ft is not None else "nao atingido"
        mg=f"{pe-ft} dB" if pe is not None and ft is not None else "-"
        ls.append(f"| {k} | {pe_s} | {ft_s} | {mg} |")
    return "\n".join(ls)

md="# DADOS LEAN (120 bits, 11 niveis SNR) — pronto para relatorio\n\n"
md+=tabela(resA,snrs,"Exp A — padrao (0.5s | 440/880 Hz)")+"\n\n"
md+=tabela(resB1,snrs,"Exp B1 — rapido + banda estreita (0.1s | 440/550 Hz)")+"\n\n"
md+=tabela(resB2,snrs,"Exp B2 — afastado (0.5s | 300/1200 Hz)")+"\n\n"
Path(OUT/"DADOS_RELATORIO_LEAN.md").write_text(md,encoding="utf-8")
print("\n"+md)
# mata o processo antigo pendurado se houver
