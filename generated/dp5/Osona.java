package model;

import model.Racun;

public class Osona {

	private String me;
	private String prezime;
	private Collection<Racun> racunCollection;

	public Osona () { }

	public Osona (String me, String prezime, Collection<Racun> racunCollection) {
		this.me = me;
		this.prezime = prezime;
		this.racunCollection = racunCollection;
	}


	public void ispisi ( ) {
		return null;
	}

}
