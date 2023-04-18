package Vue;

import Controleur.Control;
import Controleur.Voler;
import Modele.Etat;
import Modele.Parcours;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.util.ArrayList;


/** Affichage
 *La classe Affichage hérite de JPanel. Elle dispose d’un attribut de type Etat pour accéder à l’état du modèle lorsque
 * l’affichage doit être revu, via la méthode paint. Elle dispose aussi de l’ensemble des constantes liées à l’interface
 * graphique (taille de la fenêtre, dimensions des objets à dessiner, etc).
 */


public class Affichage extends JPanel implements KeyListener {
	public Etat etat;
	private Control control;
	private Voler voler;
	public Parcours p;
	public static final int LARGEUR_FENETRE = 800;
	public static final int HAUTEUR_FENETRE = 800;
	public static final int LARG_OVAL = 20;
	public static final int HAUT_OVAL = 100;

	public Affichage() {
		etat = new Etat(this);
		this.setPreferredSize(new Dimension(LARGEUR_FENETRE, HAUTEUR_FENETRE));
		etat.setMax(HAUTEUR_FENETRE - HAUT_OVAL);
		etat.startParcours(LARGEUR_FENETRE, etat.getHauteur());
		this.addKeyListener(this);
		control = new Control(etat, this);
		voler = new Voler(etat,this);
		voler.start();
		this.addMouseListener(control);
		this.addKeyListener(control);

	}

	/**
	 * Getter de la largeur de la fenêtre
	 * @param /
	 * @return int
	 */


	public static int getWIDTH() {
		return LARGEUR_FENETRE;
	}




	/**
	 * Permet la création d'un oval (et début d'un oiseau et du parcours)
	 * @param g
	 * @return void
	 */


	@Override
	public void paint(Graphics g) {
		super.paint(g);
		Graphics2D g2 = (Graphics2D) g;
		g2.setFont(new Font("Arial", Font.PLAIN, 32));
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		int hauteur = etat.getHauteur();
		g2.setStroke(new BasicStroke(3.0f));
		g2.setColor(Color.BLACK);
		g2.drawOval(Etat.X_OVAL, hauteur - HAUT_OVAL/2, LARG_OVAL, HAUT_OVAL);
		g.drawString("Score : " + etat.parcours.getPosition() , getWIDTH()-200 , 50);

		if (etat.getPerdu() == false) {
			g2.setColor(Color.BLUE);
			g2.drawString("Perdu", 0, 32);
		}
 
		g2.setColor(Color.RED);
		ArrayList<Point> points = etat.getParcours();
		Point last_point = points.get(0);
		for (int i = 0; i < points.size() - 1; i += 1) {
			Point p1 = points.get(i);
			Point p2 = points.get(i + 1);
			g2.drawLine(p1.x, p1.y, p2.x, p2.y);
			last_point = p2;
		}
	}
	
	public void affichageFinPartie() {
		System.out.println("aff fin");
        JOptionPane.showMessageDialog(this,"Votre Score : " + etat.parcours.getPosition()/2, "FIN DE PARTIE", JOptionPane.PLAIN_MESSAGE);
        System.exit(0);
	}
	/**
	 * Permet l'actualisation de la fenêtre (déplacement de l'oval)
	 * @param /
	 * @return void
	 */


	public void change() {
		repaint();
		revalidate();
	}

	/**
	 * Méthodes des differentes interactions avec souris et clavier
	 * @param e
	 * @return void
	 * @return string Hi
	 */


	@Override
	public void keyTyped(KeyEvent e) {
		// TODO Auto-generated method stub
	}

	@Override
	public void keyPressed(KeyEvent e) {
		System.out.println("Hi");
	}

	@Override
	public void keyReleased(KeyEvent e) {
		// TODO Auto-generated method stub
	}
}