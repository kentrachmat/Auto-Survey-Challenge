/**
 * Write a description of class Lightbulb here.
 *
 * @author (Benedictus Kent Rachmat)
 * @version (21-9-2020)
 */
public class Lightbulb
{
    private boolean on;
    private String color;
    private int power;

    /**
     * Constructor for objects of class Lightbulb
     */
    public Lightbulb(String lights)
    { 
        this.color = "White";
        this.power = 1;
        this.on = false;
    }

    /**
     * Display the condition of a bulb
     * @return boolean
     */
    public boolean isOn()
    {
        return on;
    }
    
    /**
     * Display the power of a bulb
     * @return power
     */
    public int getPower()
    {
        return power;
    }
    
    /**
     * Display the color of a bulb
     * @return color
     */
    public String getColor()
    {
        return color;
    }
    
    /**
     * turn this bulb on
     */
    public boolean on() 
    {
        return true;
    }

    /**
     * turn this bulb off
     */
    public boolean off() {
        return false;
    }

    /**
     * Display name and value of goods
     * @return sentences of information
     */
    public String toString()
    {
        String i;
        if(on ==false){i="off";}
        else{i="on";}
        return "Power: "+power+"W, Color: "+color+", and it is "+i+".";
    }
}