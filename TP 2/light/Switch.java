/**
 * Write a description of class Lightbulb here.
 *
 * @author (Benedictus Kent Rachmat)
 * @version (21-9-2020)
 */
public class Switch
{
    private String light;
    Lightbulb ons = new Lightbulb(light);
    /**
     * Constructor for objects of class Switch
     */
    public Switch()
    {
        //this.on =false;
    }
    
    /**
     * turn this bulb on
     */
    public void on() 
    {
       this.ons.on();
    }

    /**
     * turn this bulb off
     */
    public void off() {
        this.ons.off();
    }

}
