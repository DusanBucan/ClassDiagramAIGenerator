package model;

import model.Racun;

public class Banka {

	public String ime;
	private String adrasa;
	private Collection<Racun> racunCollection;

	public Banka () { }

	public Banka (String ime, String adrasa, Collection<Racun> racunCollection) {
		this.ime = ime;
		this.adrasa = adrasa;
		this.racunCollection = racunCollection;
	}


	public String bankaDetalji ( ) {
		return null;
	}

}
