import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, pathlib, csv
OUT = pathlib.Path(r"C:\Users\UsoPessoal\Desktop\projects\REDES\lab2-real\rc\lab_integrado_unidade01\modulacao")
SR=44100

def gen(f,d,sr=SR):
    n=int(sr*d)
    t=np.linspace(0,d,n,endpoint=False)
    return np.sin(2*np.pi*f*t)*np.hanning(n)
def det(seg,sr=SR):
    fft=np.fft.fft(seg)
    freqs=np.fft.fftfreq(len(fft),1/sr)
    mag=np.abs(fft[:len(fft)//2])
    fp=freqs[:len(freqs)//2]
    return abs(fp[np.argmax(mag)])
def b2b(f,thr): return '1' if f>thr else '0'
def noise(sig,snr_db):
    sp=np.mean(sig**2)
    snr=10**(snr_db/10)
    pw=sp/snr
    return sig+np.random.normal(0,np.sqrt(pw),len(sig))

def enc_nrz(bits,bd,fl,fh,sr): return np.concatenate([gen(fh if b=='1' else fl,bd,sr) for b in bits])
def dec_nrz(sig,n,bd,fl,fh,sr):
    thr=(fl+fh)/2; spb=int(sr*bd); o=[]
    for i in range(n):
        s,e=i*spb,(i+1)*spb
        if e>len(sig): break
        o.append(b2b(det(sig[s+spb//4:e-spb//4],sr),thr))
    return ''.join(o)
def enc_man(bits,bd,fl,fh,sr):
    h=bd/2; segs=[]
    for b in bits:
        if b=='1': segs+=[gen(fh,h,sr),gen(fl,h,sr)]
        else: segs+=[gen(fl,h,sr),gen(fh,h,sr)]
    return np.concatenate(segs)
def dec_man(sig,n,bd,fl,fh,sr):
    thr=(fl+fh)/2; spb=int(sr*bd); o=[]
    for i in range(n):
        s,e=i*spb,(i+1)*spb
        if e>len(sig): break
        mid=s+spb//2; pad=spb//8
        b1=b2b(det(sig[s+pad:mid-pad],sr),thr); b2=b2b(det(sig[mid+pad:e-pad],sr),thr)
        o.append('1' if b1=='1' and b2=='0' else '0' if b1=='0' and b2=='1' else '?')
    return ''.join(o)
def enc_nrzi(bits,bd,fl,fh,sr):
    cur=fl; segs=[]
    for b in bits:
        if b=='1': cur=fh if cur==fl else fl
        segs.append(gen(cur,bd,sr))
    return np.concatenate(segs)
def dec_nrzi(sig,n,bd,fl,fh,sr):
    thr=(fl+fh)/2; spb=int(sr*bd); o=[]; prev='0'
    for i in range(n):
        s,e=i*spb,(i+1)*spb
        if e>len(sig): break
        cur=b2b(det(sig[s+spb//4:e-spb//4],sr),thr)
        o.append('1' if cur!=prev else '0'); prev=cur
    return ''.join(o)
def ber(o,d): return (sum(1 for a,b in zip(o,d) if a!=b)+abs(len(o)-len(d)))/len(o) if o else 0

def experimento(bits,snrs,bd,fl,fh,sr,trials=8):
    res={'NRZ':[],'Manchester':[],'NRZI':[]}; n=len(bits)
    for snr in snrs:
        b={'NRZ':[],'Manchester':[],'NRZI':[]}
        for _ in range(trials):
            b['NRZ'].append(ber(bits, dec_nrz(noise(enc_nrz(bits,bd,fl,fh,sr),snr),n,bd,fl,fh,sr)))
            b['Manchester'].append(ber(bits, dec_man(noise(enc_man(bits,bd,fl,fh,sr),snr),n,bd,fl,fh,sr)))
            b['NRZI'].append(ber(bits, dec_nrzi(noise(enc_nrzi(bits,bd,fl,fh,sr),snr),n,bd,fl,fh,sr)))
        for k in res: res[k].append(float(np.mean(b[k])))
        print(f"  {snr:>4} dB -> NRZ {res['NRZ'][-1]:.3f} MAN {res['Manchester'][-1]:.3f} NRZI {res['NRZI'][-1]:.3f}")
    return res

def plot_res(res,snrs,titulo,fn):
    plt.figure(figsize=(10,5))
    cols={'NRZ':'#1f77b4','Manchester':'#ff7f0e','NRZI':'#2ca02c'}
    for k,v in res.items(): plt.plot(snrs,v,marker='o',ms=4,lw=1.6,label=k,color=cols[k])
    plt.xlabel('SNR (dB) — canal piora para a direita'); plt.ylabel('BER'); plt.title(titulo,fontsize=11)
    plt.ylim(-0.02,0.62); plt.grid(True,alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(OUT/fn,dpi=150); plt.close(); print(f"saved {fn}")

def salvar_csv(res,snrs,fn):
    with open(OUT/fn,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['SNR_dB','NRZ','Manchester','NRZI'])
        for i,s in enumerate(snrs): w.writerow([s,res['NRZ'][i],res['Manchester'][i],res['NRZI'][i]])
    print(f"saved {fn}")

def tabela_md(res,snrs,titulo):
    ls=[f"### {titulo}","", "| Modulacao | 1o erro (BER>0) | Falha total (BER>=0.5) | Margem |","|:---|:---:|:---:|:---:|"]
    for k,v in res.items():
        pe=next((snrs[i] for i,x in enumerate(v) if x>0),None)
        ft=next((snrs[i] for i,x in enumerate(v) if x>=0.5),None)
        pe_s=f"{pe} dB" if pe is not None else "nao atingido"
        ft_s=f"{ft} dB" if ft is not None else "nao atingido"
        mg=f"{pe-ft} dB" if pe is not None and ft is not None else "-"
        ls.append(f"| {k} | {pe_s} | {ft_s} | {mg} |")
    return "\n".join(ls)

np.random.seed(7)
bits=''.join(np.random.choice(list('01'),200))
snrs=np.arange(30,-36,-2)  # 30 .. -34 (33 niveis) - estendido
print(f"bits={len(bits)} snrs {snrs[0]}..{snrs[-1]} ({len(snrs)} niveis)")

print("\n=== EXT A padrao (0.5s 440/880 44100) ===")
resA=experimento(bits,snrs,0.5,440,880,44100,trials=5)
plot_res(resA,snrs,"Q1.1 EXT — A padrao (0.5s | 440/880 Hz | 44100) | SNR x BER","q11_EXT_A.png")
salvar_csv(resA,snrs,"q11_EXT_A.csv")

print("\n=== EXT B1 pior caso (0.05s 440/500 8000) ===")
resB1=experimento(bits,snrs,0.05,440,500,8000,trials=5)
plot_res(resB1,snrs,"Q1.1 EXT — B1 pior caso (0.05s | 440/500 Hz | 8000) | SNR x BER","q11_EXT_B1.png")
salvar_csv(resB1,snrs,"q11_EXT_B1.csv")

print("\n=== EXT B2 banda larga (0.05s 300/1200 44100) ===")
resB2=experimento(bits,snrs,0.05,300,1200,44100,trials=5)
plot_res(resB2,snrs,"Q1.1 EXT — B2 banda larga (0.05s | 300/1200 Hz) | SNR x BER","q11_EXT_B2.png")
salvar_csv(resB2,snrs,"q11_EXT_B2.csv")

print("\n=== EXT B3 intermediario (0.1s 440/550 44100) ===")
resB3=experimento(bits,snrs,0.1,440,550,44100,trials=5)
plot_res(resB3,snrs,"Q1.1 EXT — B3 (0.1s | 440/550 Hz) | SNR x BER","q11_EXT_B3.png")
salvar_csv(resB3,snrs,"q11_EXT_B3.csv")

for label,res in [("EXT A",resA),("EXT B1 pior",resB1),("EXT B2 larga",resB2),("EXT B3",resB3)]:
    print(f"\n-- {label} --")
    for k,v in res.items():
        pe=next((snrs[i] for i,x in enumerate(v) if x>0),None)
        ft=next((snrs[i] for i,x in enumerate(v) if x>=0.5),None)
        print(f"{k:<12} 1o {pe}  falha {ft}")

md="# DADOS EXTENDIDOS (200 bits, SNR 30..-34 step 2, trials 5)\n\n"
md+=tabela_md(resA,snrs,"Exp A — padrao (0.5s | 440/880 Hz | 44100)")+"\n\n"
md+=tabela_md(resB1,snrs,"Exp B1 — pior caso (0.05s | 440/500 Hz | 8000)")+"\n\n"
md+=tabela_md(resB2,snrs,"Exp B2 — banda larga (0.05s | 300/1200 Hz | 44100)")+"\n\n"
md+=tabela_md(resB3,snrs,"Exp B3 — intermediario (0.1s | 440/550 Hz | 44100)")+"\n\n"
md+="Arquivos: q11_EXT_*.png e .csv\n"
pathlib.Path(OUT/"DADOS_RELATORIO_EXTENDED.md").write_text(md,encoding="utf-8")
print("\n"+md)
