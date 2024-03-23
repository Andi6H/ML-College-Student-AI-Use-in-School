# College Student AI Use in School

## Informacioni i Universitetit
- **Universiteti**: Hasan Prishtina
- **Fakulteti**: Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike - FIEK
- **Niveli Akademik**: Master
- **Lënda**: Machine Learning
- **Mësimdhënësit**: Lule Ahmedi dhe Mërgim Hoti

## Studentët që kanë kontribuar në projekt
- Andi Hyseni
- Albin Bajrami

## Faza 1: Përgatitja e modelit

### Hapat e ekzekutimit:
Për të ekzekutuar këtë fazë të projektit, fillimisht duhet të instaloni librarinë pandas dhe scikit-learn duke shkruar në terminal:
```
pip install pandas
pip install scikit-learn
```

Është e rekomandueshme të krijojmë një ambient virtual (venv) për projektin për të izoluar librarinë dhe versionet e Python që përdorim. Për të krijuar një venv, shkruani në terminal:
```
python -m venv myenv
```
Kjo do të krijojë një ambient virtual të quajtur "myenv". Mund të ndryshoni "myenv" me emrin që preferoni.

Pasi keni krijuar venv, duhet ta aktivizoni atë. Në sistemet Windows, shkruani në terminal:
```
myenv\Scripts\activate
```
Në sistemet Linux/MacOS, shkruani:
```
source myenv/bin/activate
```

Pas instalimit të librave dhe aktivizimit të ambientit virtual, mund të ekzekutoni skedarin main.ipynb. Për ta ekzekutuar, hapni një terminal në direktoriumin e projektit dhe shkruani:
```
jupyter notebook main.ipynb
```

### Detajet e datasetit:
Në këtë projekt ne kemi përdorur një dataset të huazuar nga Kaggle në linkun në vijim: [College Student AI Use in School](#https://www.kaggle.com/datasets/trippinglettuce/college-student-ai-use-in-school/data), i cili përmban 258 rreshta dhe 7 kolona:
```
- Timestamp
- On a scale from 1 to 5, how would you rate your knowledge and understanding of Artificial Intelligence (AI)?
- On a scale from 1 to 5, how often do you use Artificial Intelligence (AI) for personal use?
- On a scale from 1 to 5, how often do you use Artificial Intelligence (AI) for school-related tasks?
- On a scale from 1 to 5, how interested are you in pursuing a career in Artificial Intelligence?
- Do you know what Chat-GPT is?
- What college are you in?
```
