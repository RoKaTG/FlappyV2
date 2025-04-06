\section{Convolution temporelle et fréquentielle}

\subsection{Convolution discrète classique}
La convolution d’un signal \( s[n] \) avec un filtre \( h[k] \) s’écrit :
\begin{equation}
y[n] = \sum_{k=0}^{M-1} s[n-k] \cdot h[k]
\end{equation}
Complexité : \( \mathcal{O}(N_s \cdot M) \)

\subsection{Convolution via le théorème de convolution (fréquentiel)}
En transformant dans le domaine de Fourier :
\begin{equation}
y[n] = \text{FT}^{-1} \left( \text{FT}(h) \cdot \text{FT}(s) \right)
\end{equation}
La complexité devient \( \mathcal{O}(N \log N) \), où \( N \geq N_s + M - 1 \).

\section{Méthode Overlap-and-Save (OLS)}

\subsection{Principe}
Le signal est découpé en segments qui se recouvrent. Chaque segment est traité indépendamment :

\begin{enumerate}
    \item Découpage du signal en segments de taille \( N \)
    \item Application de la FFT
    \item Multiplication élément par élément avec le filtre dans le domaine fréquentiel
    \item Application de l’inverse FFT
    \item Suppression des échantillons aliasés (de bord)
    \item Fusion des segments nettoyés dans le signal de sortie
\end{enumerate}

Le nombre d’échantillons valides par segment est : 
\[
L = N - M + 1
\]

\section{Optimisation sur GPU}

\subsection{Limites des bibliothèques standards}
Les solutions cuFFT/cuDNN nécessitent des transferts mémoire coûteux entre mémoire globale et partagée. L’article propose d’exécuter l’intégralité de l’OLS en mémoire partagée.

\subsection{Implémentation SM-OLS (Shared Memory OLS)}
Deux variantes :
\begin{itemize}
    \item \textbf{C2C (complex-to-complex)} avec FFT de Cooley-Tukey (sans reordering)
    \item \textbf{R2R (real-to-real)} avec FFT de Stockham (auto-sort)
\end{itemize}

Chaque bloc de threads GPU :
\begin{itemize}
    \item lit un segment
    \item applique une FFT en mémoire partagée
    \item effectue les multiplications complexes
    \item applique l’inverse FFT
    \item supprime les zones aliasées
    \item écrit la sortie filtrée
\end{itemize}
