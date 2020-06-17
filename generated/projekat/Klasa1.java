package model;

import model.Klasa2;
import model.Klasa3;

public class Klasa1 {

	private Collection<Klasa2> klasa2Collection;
	private Collection<Klasa3> klasa3Collection;
	public Klasa1 () { }

	public Klasa1 (Collection<Klasa2> klasa2Collection, Collection<Klasa3> klasa3Collection) {
		this.klasa2Collection = klasa2Collection;
		this.klasa3Collection = klasa3Collection;
	}


}
