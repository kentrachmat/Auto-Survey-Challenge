
/**
 * Write a description of class LightString here.
 *
 * @author (Benedictus Kent Rachmat)
 * @version (21-9-2020)
 */
public class LightString
{
    private Lightbulb[] light;
    private static final int max = 10;
    private int n;
    private String lights;
    

    /**
     * Constructor for objects of class LightString
     */
    public LightString(String lights) {
        this.light = new Lightbulb[max];
        this.n = 0;
        this.lights = lights;
        for(int i = 0; i < max;i++)
        {
        Lightbulb lightbulbs = new Lightbulb(this.lights);
        this.light[this.n] = lightbulbs;
	this.n = this.n + 1;
        }
    }
    
    /**
     * get lamp from Lightstring
     * @param n => int
     */
    public void getLight(int n)
    {
        if (n < max){
        System.out.println(this.light[n].toString());
        }
        else{
        System.out.println("Error");
        }
    }
    
    /**
     * turn all bulb on
     */
    /*public void on() 
    {   int i=0;
        while(i<n)
        {
        //this.light[i] = ons.on();
        i++;
        }
    }*/

    /**
     * turn all bulb off
     */
    public void off() 
    {   int i=0;
        while(i<n)
        {
       // this.light[i] = ons.off();
        i++;
        }
    }
    
    
    /** replace the n-th lightbulb of the light string by the given lightbulb. 
     * Nothing happens if i is not a valid index.
     * @param i the number of the lightbulb to be changed (first has number 1) 
     * @param theBulb the new lightbulb
     */
    public void changeLightbulb(int i, Lightbulb theBulb)
    {
    }

    /**
     * Get all the lights power ( turned on only )
     */
    public void getConsumedPower()
    {
    }
}
