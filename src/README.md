\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[french]{babel}
\usepackage{amsmath, amssymb}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{mathrsfs}
\usepackage{enumitem}
\geometry{margin=2.5cm}

\title{Traitement Radar : Impulsion, Filtrage Doppler et Techniques de Convolution}
\author{}
\date{}

\begin{document}

\maketitle

\section{Résolution en distance et forme de l'impulsion}

Une \textbf{impulsion courte} permet une bonne résolution en distance, ce qui facilite la distinction entre des cibles proches. Toutefois, comme elle contient peu d'énergie, le signal reçu est plus affecté par le bruit.

\bigskip

Pour pallier ce problème, on émet une \textbf{impulsion plus longue}, généralement modulée en fréquence (chirp) ou en phase. À la réception, on applique un \textbf{filtre de compression} (filtre adapté) qui permet de retrouver la résolution d’une impulsion courte tout en conservant les bénéfices énergétiques d’une impulsion longue.

\section{Traitement par convolution}

Comme la forme de l’impulsion émise est connue, on peut appliquer une \textbf{convolution temporelle} entre le signal reçu et le filtre adapté (souvent la version conjuguée et inversée du signal émis) :

\begin{equation}
y(t) = \int_{-\infty}^{+\infty} x(\tau) \cdot h(t - \tau) \, d\tau
\end{equation}

On peut également réaliser cette opération dans le domaine fréquentiel grâce à la \textbf{transformée de Fourier}, en exploitant la propriété suivante :

\begin{equation}
Y(f) = X(f) \cdot H(f)
\end{equation}

\section{Filtrage Doppler}

Si la cible est en mouvement, on observe un \textbf{décalage de fréquence} (effet Doppler), ce qui se manifeste par une variation de phase dans le signal reçu. Cette information permet d’estimer la \textbf{vitesse radiale} de la cible.

\bigskip

Pour cela, on émet une série de \( N \) impulsions espacées régulièrement. Pour chaque distance (ou \textit{range bin}), on construit une séquence temporelle à partir des retours successifs. On applique ensuite une \textbf{transformée de Fourier discrète (DFT)} :

\begin{equation}
X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-j2\pi kn/N}
\end{equation}

Cela permet d’identifier les composantes fréquentielles associées aux vitesses des cibles.

\section{Techniques et filtres utilisés}

Voici différentes techniques utilisées pour analyser les signaux radar :

\begin{itemize}[label=--]
    \item \textbf{Filtre passe-bande} : pour extraire une bande de fréquences spécifiques.
    
    \item \textbf{Short-Time Fourier Transform (STFT)} : utilisée pour observer l’évolution fréquentielle dans le temps.
    \begin{equation}
    STFT\{x(t)\}(m, \omega) = \int_{-\infty}^{+\infty} x(t)w(t - m) e^{-j\omega t} \, dt
    \end{equation}

    \item \textbf{Transformée de Hilbert} : permet de construire l’enveloppe analytique d’un signal.
    \begin{equation}
    \hat{x}(t) = \frac{1}{\pi} \, \text{p.v.} \int_{-\infty}^{+\infty} \frac{x(\tau)}{t - \tau} \, d\tau
    \end{equation}

    \item \textbf{Filtre adapté (Matched Filter)} : maximise le rapport signal sur bruit (SNR) pour un signal connu. C’est une technique clé en radar pour détecter la présence d’une cible.
\end{itemize}

\section{Convolution et filtrage}

La \textbf{convolution} est une opération fondamentale dans le traitement des signaux. Elle consiste à "faire glisser" un signal sur un autre, avec inversion temporelle de l’un des deux. Mathématiquement, la convolution entre deux signaux \( x(t) \) et \( h(t) \) est définie par :

\begin{equation}
y(t) = (x * h)(t) = \int_{-\infty}^{+\infty} x(\tau) \cdot h(t - \tau) \, d\tau
\end{equation}

Dans le cadre des systèmes radar, la convolution est utilisée entre le signal reçu et un \textbf{filtre adapté} (ou matched filter), dont la forme correspond à celle de l’impulsion émise (renversée et conjuguée dans le cas complexe). Cette opération permet :

\begin{itemize}[label=--]
    \item d’augmenter le \textbf{rapport signal sur bruit} (SNR),
    \item de faciliter la \textbf{détection} des cibles,
    \item d’extraire les composantes fréquentielles d’intérêt,
    \item et d’obtenir une meilleure \textbf{résolution temporelle}.
\end{itemize}

En pratique, la convolution est souvent effectuée de manière plus efficace dans le domaine fréquentiel grâce à la propriété :

\[
x(t) * h(t) \xrightarrow{\mathcal{F}} X(f) \cdot H(f)
\]

Ce qui permet de transformer la convolution en une multiplication simple dans le domaine de Fourier.

\section{Traitement fréquentiel par FFT}

Pour améliorer l’efficacité du traitement, on utilise la \textbf{convolution fréquentielle} à l’aide de la \textbf{Transformée de Fourier Rapide (FFT)}.

Plutôt que de réaliser la convolution dans le domaine temporel, on applique la propriété suivante :

\begin{enumerate}
    \item Calcul de la transformée de Fourier du signal reçu : \( X(f) = \mathcal{F}\{x(t)\} \)
    \item Calcul de la transformée de Fourier du filtre : \( H(f) = \mathcal{F}\{h(t)\} \)
    \item Multiplication des deux spectres : \( Y(f) = X(f) \cdot H(f) \)
    \item Application de la transformée de Fourier inverse : \( y(t) = \mathcal{F}^{-1}\{Y(f)\} \)
\end{enumerate}

Cette méthode est particulièrement efficace pour les signaux longs, et elle réduit le coût de calcul comparé à la convolution directe.

\section{Filterbank (Banques de filtres)}

Une \textbf{banque de filtres (filterbank)} est un ensemble de filtres qui décomposent un signal en plusieurs bandes de fréquences. En radar, on l’utilise par exemple pour créer une \textbf{filterbank Doppler}, c’est-à-dire une série de filtres centrés sur différentes fréquences Doppler.

Chaque filtre est conçu pour répondre à une fréquence spécifique, ce qui permet de détecter différentes \textbf{vitesses radiales de cibles}.

\bigskip

Cette approche est souvent utilisée en parallèle pour analyser plusieurs hypothèses de vitesse ou pour augmenter la résolution fréquentielle.

\section{Log-compression (LogMod)}

Après filtrage, le signal peut présenter une très grande dynamique d’amplitude ou de puissance. Pour faciliter son affichage et son traitement, on applique une \textbf{compression logarithmique (log-modulation)} :

\begin{equation}
x_{\text{log}}(t) = \log\left(1 + \alpha \cdot |x(t)|^2 \right)
\end{equation}

où \( \alpha \) est un facteur d’échelle.

\bigskip

Cette opération rend les cibles plus visibles, notamment les plus faibles, et facilite la détection automatique (par exemple avec le \textbf{CFAR} – \textit{Constant False Alarm Rate}).

\section{Padding (bourrage de zéros)}

Lorsqu'on applique une transformée de Fourier (via FFT), il est souvent nécessaire de \textbf{faire du padding}, c'est-à-dire d'ajouter des zéros à la fin du signal. Cela a plusieurs avantages :

\begin{itemize}[label=--]
    \item \textbf{Aligner la taille du signal} : pour correspondre à la taille d'une fenêtre de traitement ou à une puissance de 2 (optimale pour la FFT).
    \item \textbf{Éviter les effets de bord} : limiter les artefacts dus à des coupures brutales en fin de signal.
    \item \textbf{Améliorer la résolution fréquentielle} : un padding important augmente le nombre de points FFT, ce qui affine l’analyse spectrale.
\end{itemize}

