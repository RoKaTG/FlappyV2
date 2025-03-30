---

## 1. La Pulse Compression (PC)

**Contexte :**  
En radar, lorsqu’on émet une impulsion courte, on obtient une bonne résolution en distance (on peut « distinguer » des cibles proches). Mais émettre une impulsion très courte signifie émettre moins d’énergie, donc avoir une portée plus faible et un rapport signal-sur-bruit potentiellement moins bon.  
Pour contourner cela, on peut émettre une impulsion *plus longue* (plus d’énergie), mais qu’on module (en fréquence ou en phase) afin qu’au moment de la réception, on applique un filtre adapté (une « compression » de l’impulsion). Cela permet de combiner l’avantage d’un signal de longue durée (plus d’énergie) et d’une impulsion « efficacement courte » après traitement (bonne résolution).

**Principe simplifié :**  
1. Le radar émet un signal modulé (par exemple une « chirp » linéaire, ou un code à phase pseudo-aléatoire).  
2. À la réception, on applique un *matched filter* (filtre adapté) qui fait la *corrélation* du signal reçu avec la forme connue de l’impulsion émise.  
3. Cette corrélation aboutit à un pic bref (impulsion comprimée) qui améliore la résolution en distance.

**En pratique :**  
La compression d’impulsion est souvent réalisée par une **convolution** du signal reçu avec le *filtre adapté*, qui est généralement la réplique conjuguée et inversée dans le temps du signal émis. Dans le domaine fréquentiel, on fait cela grâce à des FFT (transformation de Fourier rapide) :  
\[
\text{PC}(t) = \mathcal{F}^{-1} \Big(\mathcal{F}(\text{signal reçu}) \times \mathcal{F}(\text{filtre adapté})\Big).
\]

---

## 2. Le Doppler Filtering (DF)

**Contexte :**  
Les signaux radar, quand ils se réfléchissent sur des cibles en mouvement, sont décalés en fréquence (effet Doppler). On peut utiliser cette information pour estimer la vitesse radiale de la cible.  
En radar à impulsions, on effectue souvent une succession d’impulsions (une rafale de N impulsions). Ensuite, on traite l’évolution des signaux reçus d’une impulsion à l’autre.

**Principe simplifié :**  
1. On considère la trame temporelle d’échos reçus à chaque « coup » radar (chaque impulsion émise).  
2. Pour chaque cellule de distance (range bin), on prend la suite temporelle des retours sur N impulsions, ce qui forme un petit signal dans le temps (indexé par le numéro d’impulsion).  
3. On applique une transformée de Fourier sur ces N points pour obtenir les fréquences Doppler (i.e., la vitesse radiale).  

Cette transformée de Fourier discrète sur l’axe « impulsion » est souvent appelée « **Doppler filtering** » ou « FFT Doppler » (ou « DFT »). Résultat : on obtient un spectre Doppler pour chaque range bin.

---

## 3. La Convolution sur signal radar

**Définition :**  
La convolution est une opération mathématique définie par :  
\[
y[n] = \sum_{k=-\infty}^{+\infty} x[k] \, h[n-k]
\]  
où \(x\) est le signal d’entrée et \(h\) le filtre (ou réponse impulsionnelle).  

**Dans le domaine radar :**  
- Pour la compression d’impulsion, on fait typiquement la convolution entre le signal reçu et le filtre adapté.  
- Pour éliminer du bruit ou extraire certaines composantes fréquentielles, on peut aussi réaliser d’autres filtrages par convolution.

**Remarque :**  
En pratique, pour des raisons d’effica, on n’effectue pas forcément la convolution dans le domaine temporel, mais plutôt dans le domaine fréquentiel (via FFT). Dans du code C/C++/CUDA, vous trouverez parfois la démarche suivante :  
1. On calcule la FFT du signal.  
2. On calcule la FFT de la réponse impulsionnelle du filtre.  
3. On multiplie les deux dans le domaine fréquentiel.  
4. On applique la transformée de Fourier inverse (IFFT) pour revenir dans le domaine temporel.  

Cela revient mathématiquement au même que la convolution directe, mais c’est plus rapide (surtout pour de longues impulsions ou pour un grand nombre de points).

---

## 4. C’est quoi un Filter Bank (Banque de filtres) ?

Une *filter bank* (banque de filtres) est un ensemble de plusieurs filtres (généralement en parallèle) qui décomposent un signal en différentes bandes de fréquences ou canaux.  

**Dans le cadre radar (exemples) :**  
- On peut utiliser une banque de filtres Doppler (chacun accordé sur une fréquence Doppler particulière) pour distinguer plusieurs vitesses cibles.  
- On peut aussi décomposer le spectre d’un signal sur plusieurs sous-bandes pour faire de la détection plus fine ou de l’estimation paramétrique.  

C’est donc un outil pour analyser le contenu fréquentiel (ou dans d’autres domaines, comme temps-fréquence) de manière plus précise ou plus adaptative qu’une simple transformée de Fourier.

---

# Illustration par pseudo-code GPU (exemple simplifié)

Pour illustrer le principe, voici un *pseudo-code* en style CUDA (très simplifié) qui montre comment on pourrait faire une convolution ou un filtrage Doppler sur GPU.  
   
**Scénario :**  
- On suppose qu’on a un signal radar « `inputSignal` » de longueur `N`.  
- On veut le convoluer avec un filtre « `filter` » de longueur `M`.  
- Le résultat fait `N + M - 1` points.  
- On fera une version « naïve » en temps (sans FFT) pour la pédagogie, puis on esquisse la version FFT.  
- On donne juste des grandes lignes, sans rentrer dans les détails de l’allocation mémoire GPU, la gestion des blocs, etc.

### A) Convolution directe naïve sur GPU

```cpp
// Nombre de threads total = N + M - 1 (un thread par élément de la sortie)
// Hypothèse : On gère la répartition en blocks et threads GPU plus bas niveau.

__global__ void convolveNaiveGPU(const float* inputSignal, int N,
                                 const float* filter, int M,
                                 float* outputSignal)
{
    // Calcul de l’index global du thread
    int outIndex = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (outIndex < (N + M - 1)) {
        float sum = 0.0f;
        // Effectuer la somme sur le domaine de convolution
        for (int k = 0; k < M; k++) {
            int inIndex = outIndex - k;
            // Vérifier qu’on est dans les bornes
            if (inIndex >= 0 && inIndex < N) {
                sum += inputSignal[inIndex] * filter[k];
            }
        }
        outputSignal[outIndex] = sum;
    }
}

// ...
// Code hôte simplifié
// ...
int main() {
    // 1) Charger ou générer inputSignal et filter sur CPU
    // 2) Allouer la mémoire sur GPU (cudaMalloc(...))
    // 3) Copier inputSignal et filter sur GPU (cudaMemcpy(...))
    // 4) Lancer le kernel :

    int totalThreads = N + M - 1;
    int blockSize = 256;
    int gridSize = (totalThreads + blockSize - 1) / blockSize;
    
    convolveNaiveGPU<<<gridSize, blockSize>>>(d_input, N, d_filter, M, d_output);

    // 5) Récupérer la mémoire résultat sur CPU
    // 6) Libérer la mémoire GPU
    return 0;
}
```

**Remarque :** Cette méthode est « naïve » et peu efficace quand `N` et `M` sont grands. On la montre pour l’exemple.

---

### B) Convolution via FFT sur GPU (schéma simplifié)

Dans les librairies réelles, on utiliserait cuFFT (sous CUDA) ou clFFT (sous OpenCL) pour faire les FFT. Le pseudo-code :

```cpp
// Pseudo-code pour montrer l’enchaînement, en admettant qu’on dispose de routines GPU de FFT

// 1) Zero-pad le signal d’entrée et le filtre pour que leur taille soit de puissance de 2, >= (N+M-1).
int sizeFFT = trouverTaillePuissance2( N + M - 1 );

// 2) Copier inputSignal et filter dans des buffers "complexes" avec zero-padding
//    sur GPU : d_signalFFT, d_filterFFT

// 3) Lancer la FFT sur d_signalFFT
gpuFFTForward(d_signalFFT, sizeFFT);

// 4) Lancer la FFT sur d_filterFFT
gpuFFTForward(d_filterFFT, sizeFFT);

// 5) Effectuer la multiplication point-à-point dans le domaine fréquentiel
//    conjFilter si besoin (pour matched filtering).
multiplyComplex<<<...>>>(d_signalFFT, d_filterFFT, d_resultFFT, sizeFFT);

// 6) Lancer la FFT inverse sur d_resultFFT
gpuFFTInverse(d_resultFFT, sizeFFT);

// 7) d_resultFFT contient la convolution (ou corrélation si on utilise le conjugué).
//    Récupérer le résultat final (N+M-1) éléments sur CPU.
```

Dans le cas d’un traitement Doppler (FFT d’une rafale d’impulsions), on ferait quelque chose de similaire : on chargerait nos données radar sur GPU, puis on appellerait une FFT par « range bin » ou on ferait une FFT multidimensionnelle (2D ou 3D) :  
- 1er axe : le temps ou la distance,  
- 2e axe : le nombre d’impulsions (pour l’axe Doppler).  

---

## En résumé

- **Pulse Compression (PC)** : Technique permettant d’obtenir à la fois une haute résolution en distance (comme une impulsion courte) tout en émettant suffisamment d’énergie (impulsion longue) grâce à une modulation et à l’application d’un filtre adapté.  
- **Doppler Filtering (DF)** : Application d’une transformée de Fourier sur la séquence d’impulsions successives pour estimer les fréquences Doppler, donc détecter les vitesses radiales.  
- **Convolution** : Opération de base pour filtrer un signal, qui en radar correspond notamment à la mise en œuvre du matched filter pour la compression d’impulsion (ou d’autres filtrages).  
- **Filter Bank (banque de filtres)** : Ensemble de filtres (souvent parallèles) qui décomposent le signal sur différentes bandes ou fréquences (ou différents canaux Doppler), permettant une analyse multi-bande ou multi-vitesse.  

Pour l’**implémentation GPU**, on exploite massivement les **FFT** (avec cuFFT ou équivalent) et on parallelise les opérations de convolution/filtrage (souvent par multiplication dans le domaine fréquentiel).  

Ces principes, combinés dans les chaînes de traitement radar, permettent d’optimiser la détection des cibles, l’estimation de leur distance et vitesse, et la séparation en différents canaux (filter bank).


---

Voici un résumé synthétique des éléments clés présentés dans le document de thèse (section 5.2) concernant le traitement radar haute performance sur GPU, en mettant l’accent sur les notions de PC (Pulse Compression), DF (Doppler Filtering), convolution, FFT, logmod et le reste de la chaîne de traitement.

---

## 1. Contexte général : Traitement radar et haute performance GPU

Le document souligne l’importance d’exécuter rapidement des algorithmes de traitement radar sur de grandes quantités de données, afin de traiter en temps quasi-réel les échos provenant des impulsions successives . Les GPU (Unités de traitement graphique) se prêtent particulièrement bien à ce type de traitement massivement parallèle, car les opérations fréquentes (convolutions, FFT, filtrage Doppler…) peuvent être distribuées sur un grand nombre de cœurs.

---

## 2. Pulse Compression (PC)

### Principe
- **Objectif** : Combiner l’énergie d’une impulsion radar longue (meilleur rapport signal/bruit) avec une haute résolution en distance (comme si l’impulsion était courte).  
- **Mise en œuvre** : On module l’impulsion (en phase ou fréquence) à l’émission. Puis on applique un filtre adapté (ou « matched filter ») en réception.  
- **Implémentation** :  
  - Soit sous forme de convolution directe dans le domaine temporel entre le signal reçu et la réplique du signal émis.  
  - Soit en utilisant la **FFT** pour accélérer la convolution (multiplication dans le domaine fréquentiel suivie d’une FFT inverse).  
- **Avantage GPU** : Les GPU permettent de paralléliser les opérations de convolution ou de FFT et de traiter simultanément un grand nombre de « range bins » (cellules de distance).

---

## 3. Doppler Filtering (DF)

### Principe
- **Effet Doppler** : Les cibles en mouvement introduisent un décalage fréquentiel sur les impulsions successives.  
- **Doppler Filtering** : Sur un ensemble de N impulsions, on effectue une Transformée de Fourier (FFT) dans l’axe « pulsé » (inter-pulse), afin de détecter les composantes fréquentielles liées aux vitesses des cibles.  
- **Implémentation GPU** :  
  - Les données peuvent être réorganisées en mémoire (par « range bin »).  
  - On applique ensuite des FFT multidimensionnelles (ex. 2D : distance × Doppler) de manière parallèle.  

---

## 4. Convolution et FFT dans la chaîne radar

La **convolution** est un noyau central du traitement radar, en particulier pour la compression d’impulsion (matched filtering) et d’autres filtrages éventuels. Dans un mode « pipeline complet », on utilise souvent la FFT pour accélérer ces convolutions, et on retrouve plusieurs appels FFT/IFFT dans la chaîne (pour la PC, pour le DF, voire pour le pré-traitement ou post-traitement) .

**Optimisations GPU** :
- Utilisation de librairies de FFT optimisées (cuFFT, etc.).  
- Fusion de noyaux (kernel fusion) pour réduire les transferts mémoire (calcul direct dans le domaine fréquentiel, évitant des allers-retours multiples).

---

## 5. LogMod

Le document évoque l’étape de **LogMod**, qui consiste à appliquer une transformation logarithmique sur l’amplitude (ou la puissance) du signal après filtrage, afin de faciliter l’analyse (notamment pour la détection). Souvent, on prend la magnitude complexe (modulus), puis on passe en log pour mieux gérer les dynamiques d’amplitude très différentes. Par exemple :  
\[
\text{LogMod}(x) = 20 \log_{10}(|x|)
\]  
Cette étape est utile pour la **CFAR** (Constant False Alarm Rate) ou tout autre algorithme de détection.

---

## 6. CFAR (Constant False Alarm Rate)

Bien que vous n’ayez pas spécifiquement demandé un rappel sur la CFAR, elle est mentionnée dans la même section du pipeline. La CFAR sert à maintenir un taux de fausse alarme constant en adaptant le seuil de détection localement (en fonction du bruit ou du clutter).  
- **Implémentation GPU** : on calcule pour chaque cellule une estimation de la moyenne/variance du voisinage, puis on compare à un seuil ajusté.  
- **Lien avec LogMod** : Les données sont souvent « log-modulées » avant la CFAR, pour simplifier l’estimation et la comparaison de seuil.

---

## 7. Pipeline complet et Kernel Fusion

Le document décrit un **pipeline complet** qui enchaîne :
1. **Pulse Compression** (convolution/FFT)  
2. **Doppler Filtering** (FFT sur l’axe impulsion)  
3. **LogMod**  
4. **CFAR**  
5. (Éventuellement d’autres filtrages, détections, etc.)

Afin de **réduire la latence** et d’**augmenter le débit**, on peut employer la technique de **kernel fusion** : au lieu de lancer un kernel GPU pour chaque étape séparément, on fusionne certaines étapes afin de limiter les transferts mémoire et maximiser l’occupation du GPU. Le document mentionne cette approche comme l’une des clés pour obtenir des temps de calcul compatibles avec les applications radar temps réel .

---

## 8. Perspectives d’optimisation et de recherche

- **Augmenter la taille et la complexité des signaux** (plus de range bins, plus d’hypothèses Doppler, etc.) tout en gardant un temps de calcul réduit.  
- **Tirer parti de plusieurs GPU** (ou de GPU plus récents) pour traiter des volumes de données encore plus grands.  
- **Approfondir l’optimisation mémoire** (accès coalescés, utilisation judicieuse de la mémoire partagée) et le placement optimal des calculs FFT.  
- **Kernel fusion avancée** : combiner davantage d’opérations dans la même phase de calcul, en évitant les allers-retours inutiles en mémoire globale.

---

# Conclusion

La section 5.2 de la thèse met en évidence la façon dont la **Pulse Compression**, le **Doppler Filtering**, la **convolution**, la **FFT**, le **LogMod** et le **CFAR** s’imbriquent dans une **chaîne de traitement radar** complète, tirant profit des GPU pour exécuter ces étapes de manière parallèle et efficace . Les techniques de fusion de noyaux, de gestion optimisée de la mémoire et d’utilisation intensive de la FFT (via des bibliothèques spécialisées) sont au cœur des gains de performance pour le traitement temps réel ou quasi-temps réel.

---

Voici une version plus détaillée, avec quelques formules et explications mathématiques pour mieux comprendre chaque concept (pulse compression, Doppler filtering, convolution, FFT, logmod, etc.) dans le contexte radar et leur implémentation accélérée sur GPU .

---

## 1. Pulse Compression (PC)

### Contexte et objectif

- **Idée générale** : Émettre une impulsion radar d’assez longue durée (pour avoir suffisamment d’énergie) mais disposer d’une grande résolution en distance (comme si l’impulsion était courte).  
- **Mise en œuvre** : On module l’impulsion à l’émission (en phase ou en fréquence) puis on applique un **filtre adapté** en réception.

### Formulation mathématique

1. **Signal émis** :  
   \[
   s(t) \quad \text{(de durée } T \text{)}
   \]  
   peut être, par exemple, un **chirp linéaire** (modulation en fréquence) ou un **code binaire** (modulation en phase).  

2. **Signal reçu** (simplifié) :  
   \[
   r(t) = \alpha \, s(t - \tau) + \text{bruit}
   \]  
   où \(\tau\) est le délai dû à la distance de la cible et \(\alpha\) un coefficient d’atténuation/amplitude.

3. **Filtre adapté** :  
   Dans le domaine radar, on applique souvent le **matched filter** (filtre adapté), dont la réponse impulsionnelle \( h(t) \) est la réplique conjuguée et renversée dans le temps du signal émis :  
   \[
   h(t) = s^*(-T + t)
   \]  
   (selon la convention, on peut aussi considérer la corrélation directe, ce qui revient à peu près au même.)

4. **Compression d’impulsion** :  
   La sortie du filtre adapté se calcule comme une **convolution** (ou corrélation selon la définition) :  
   \[
   y(t) = (r \ast h)(t) = \int_{-\infty}^{\infty} r(\tau)\,h(t - \tau)\,\mathrm{d}\tau.
   \]  
   Si \(r(t)\) correspond bien à la « forme » de \(s(t)\), on obtient un pic bref, ce qui **compresse** l’impulsion dans le temps.

### Intérêt GPU

- **Convolution** et **corrélation** peuvent être réalisées efficacement via la **FFT** (voir ci-dessous).  
- Chaque « range bin » peut être traité en parallèle.  
- Les bibliothèques GPU (cuFFT, etc.) permettent d’effectuer ces opérations sur un grand nombre de points à très haute vitesse.

---

## 2. Doppler Filtering (DF)

### Contexte radar

- Lorsqu’une cible se déplace, son écho radar subit un **décalage Doppler** (changement de fréquence).  
- Dans un radar pulsé, on envoie régulièrement des impulsions (par exemple N impulsions successives). On regarde alors l’évolution de l’écho dans le temps inter-pulses.

### Formulation mathématique

1. **Signal reçu par range bin** :  
   Pour un bin de distance donné, on dispose d’un vecteur de N échantillons (un par impulsion) :  
   \[
   x = [x_0,\; x_1,\; \dots,\; x_{N-1}],
   \]  
   où \(x_m\) est la valeur (complexe) de l’écho reçu à l’impulsion \(m\).  

2. **Transformée de Fourier discrète (DFT)** :  
   \[
   X[k] = \sum_{m=0}^{N-1} x_m \, e^{-j\,\frac{2\pi}{N}\,k\,m}, \quad k = 0, \dots, N-1.
   \]  
   Chaque indice \(k\) correspond à une **fréquence Doppler** ou « vitesse » potentielle.

3. **Implémentation GPU** :  
   - On peut effectuer une **FFT 1D** pour chaque range bin.  
   - En pratique, on peut organiser les données en 2D : (distance × impulsion) et appliquer une **FFT 2D** (parfois après la Pulse Compression).  
   - Les opérations étant identiques pour chaque bin, on exploite aisément le parallélisme massif du GPU.

---

## 3. Convolution

### Définition mathématique

La **convolution** discrète de deux signaux \(x[n]\) et \(h[n]\) de longueur respective \(N\) et \(M\) est définie par :  
\[
y[n] = \sum_{k=0}^{M-1} x[n-k]\;h[k], \quad \text{avec } 0 \le n < (N+M-1),
\]  
en veillant à gérer les indices « hors limites » (on peut aussi écrire la formule sur l’intervalle complet \([- \infty, +\infty]\) et tenir compte de \(x[n] = 0\) hors de [0, N-1], etc.).

### Utilisation radar

- **Pulse Compression** : c’est la convolution du signal reçu avec le filtre adapté.  
- **Autres filtrages** : débruitage, filtrage passe-bande, etc.

### Implémentation via FFT

1. **Principe** :  
   \[
   \text{convolution}(x,h) \ \longleftrightarrow\ \text{IFFT}\Big(\text{FFT}(x)\;\times\;\text{FFT}(h)\Big),
   \]  
   où \(\times\) désigne la multiplication point-à-point dans le domaine fréquentiel.

2. **Étapes pratiques** :  
   - Zero-padding de \(x\) et \(h\) à une taille au moins égale à \(N + M - 1\).  
   - Calcul de la FFT de chaque signal.  
   - Multiplication des FFT.  
   - FFT inverse pour récupérer le résultat.  

3. **Optimisation GPU** :  
   - Les GPU sont très efficaces pour la FFT (cuFFT, etc.).  
   - On traite souvent de grands volumes de données en parallèle.

---

## 4. FFT (Fast Fourier Transform)

### Rappel

La **DFT** (Discrete Fourier Transform) d’un vecteur \(x\) de longueur \(N\) est :  
\[
X[k] = \sum_{n=0}^{N-1} x[n]\;e^{-j\,\frac{2\pi}{N}\,k\,n}, \quad k=0,\dots,N-1.
\]  
La **FFT** est un algorithme (Cooley-Tukey, etc.) qui calcule la DFT de manière plus rapide qu’une définition directe (\(\mathcal{O}(N \log N)\) au lieu de \(\mathcal{O}(N^2)\)).

### Application en radar

- **Pulse Compression** (via convolution dans le domaine fréquentiel).  
- **Doppler Filtering** (pour détecter le décalage Doppler).  
- **Filter bank** (banque de filtres) : décomposition en sous-bandes.  

### Implémentation GPU

- Les librairies spécialisées (cuFFT, clFFT, etc.) permettent d’exploiter des centaines ou milliers de cœurs GPU pour exécuter rapidement des milliers de FFT en parallèle.

---

## 5. LogMod (Logarithmic Modulus)

### Principe

Dans le traitement du signal radar, après avoir obtenu une **réponse complexe** (par exemple à la sortie du filtre adapté ou du filtrage Doppler), on s’intéresse souvent à la **magnitude** du signal. On applique alors :  
\[
\text{Mod}(z) = |z|, \quad \text{puis} \quad \text{LogMod}(z) = 20 \,\log_{10}(|z|).
\]  
- **But** : Réduire la dynamique et visualiser plus facilement les niveaux d’amplitude sur une large plage (cibles fortes vs. bruit).  
- **Utilité** : Par exemple, pour la détection (CFAR) ou pour représenter graphiquement la réponse (map Range-Doppler).

### Implémentation GPU

- Il suffit de calculer la magnitude pour chaque échantillon (opération parallèle), puis le logarithme.  
- Peut être fusionné avec d’autres kernels pour économiser les accès mémoire.

---

## 6. CFAR (Constant False Alarm Rate)

Même si vous n’avez pas explicitement demandé, le document l’évoque comme faisant partie de la chaîne de traitement.

- **Idée** : Déterminer un **seuil de détection** qui s’adapte localement au niveau de bruit/clutter afin de maintenir un taux de fausses alarmes constant.  
- **Processus** : On calcule une **statistique** sur un voisinage (moyenne, variance…), puis on compare la valeur courante à cette statistique multipliée par un facteur.  
- **LogMod** facilite cette étape en simplifiant la plage de valeurs.

---

## 7. Pipeline radar complet (rappel)

La section 5.2 de la thèse  décrit un **pipeline** typique :

1. **Lecture des données** (ou génération des signaux reçus).  
2. **Pulse Compression** (via convolution ou corrélation).  
3. **Doppler Filtering** (FFT dans l’axe impulsion).  
4. **LogMod** (pour réduire la dynamique).  
5. **CFAR** (détection en maintenant un taux de fausses alarmes constant).  
6. **Post-traitements** (extraction des cibles, etc.).

Sur GPU, on cherche à **fusionner** un maximum d’opérations (kernel fusion) et à minimiser les transferts mémoire. On profite de la parallélisation massive pour chaque étape (FFT, convolutions, calculs de magnitude…).

---

## Conclusion

Les notions de **Pulse Compression**, **Doppler Filtering**, **convolution**, **FFT**, **LogMod**, etc. sont essentielles dans le traitement radar. L’approche GPU haute performance permet de :

- Manipuler de larges volumes de données radar (multiples gammes de distance, multiples impulsions).  
- Effectuer rapidement des opérations gourmandes comme la convolution (avec ou sans FFT), la transformée de Fourier (FFT), les calculs de puissance/amplitude, et la détection.  
- Faciliter l’implémentation temps réel ou quasi temps réel en optimisant l’occupation des cœurs GPU et en réduisant les accès mémoire.
