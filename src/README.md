\section{FFT : Formulation et complexité}
La transformée de Fourier discrète (DFT) d’une séquence \( x[n] \) de taille \( N \) est donnée par :
\begin{equation}
X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j2\pi kn / N}
\end{equation}
Le calcul direct coûte \( \mathcal{O}(N^2) \). L’algorithme FFT de Cooley-Tukey réduit la complexité à \( \mathcal{O}(N \log N) \), en divisant la DFT en sous-DFT récursives.

\section{Modèle matriciel de la FFT}
La FFT est exprimée comme une suite de multiplications matricielles et de produits élémentaires :
\begin{equation}
X_{\text{out}} = F_{N_1} \cdot (T_{N_1N_2} \odot X_{\text{in}})
\end{equation}
où :
\begin{itemize}
    \item \( F_{N_1} \) : matrice DFT de taille \( N_1 \times N_1 \)
    \item \( T_{N_1N_2} \) : matrice des twiddle factors
    \item \( \odot \) : multiplication élément par élément
\end{itemize}

\section{Optimisations sur Tensor Cores}
\subsection{Limitations des APIs Tensor Core}
Les Tensor Cores sont optimisés pour les produits matriciels de type GEMM. Cependant, les FFT nécessitent :
\begin{itemize}
    \item des multiplications élémentaires complexes
    \item des accès non structurés aux matrices
\end{itemize}
Les opérations standards \texttt{mma\_sync}, \texttt{load/store\_matrix\_sync} sont limitées pour cela.

\subsection{Optimisation proposée}
L’article introduit un mappage des fragments dans les registres pour accéder à des éléments individuels :
\begin{itemize}
    \item Fragmentation contrôlée : accès aux coefficients dans les fragments
    \item Calcul des produits complexes et twiddles \textit{in situ}, sans accès mémoire partagé
\end{itemize}

\section{Architecture de tcFFT}
\begin{itemize}
    \item \textbf{Planification dynamique} : sélection de noyaux de fusion optimaux selon la taille
    \item \textbf{Support de tous les FFT en puissance de 2} (jusqu’à \(2^{23}\))
    \item \textbf{FFT 2D et batched FFT} avec accès mémoire stridés gérés efficacement
\end{itemize}

\section{Réduction du coût mémoire}
\subsection{Fusion des processus de merging}
Plusieurs étapes de merging sont combinées pour réduire les accès globaux mémoire.

\subsection{Accès coalescés en mémoire}
Réarrangement des données en mémoire pour permettre des accès alignés (in-place) avec tailles continues (jusqu’à 32) :
\begin{itemize}
    \item Permet de saturer la bande passante mémoire
    \item Réduction du besoin de synchronisation inter-blocs
\end{itemize}

\section{Équation de performance}
La performance est mesurée en TFLOPS (opérations flottantes par seconde) :
\begin{equation}
\text{TFLOPS} = \frac{6 \cdot 2 \cdot \log_2 N \cdot N \cdot N_{\text{batch}} \cdot R}{T_{\text{total}} \cdot 10^{12}}
\end{equation}
