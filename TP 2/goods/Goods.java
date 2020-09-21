/**
 * Write a description of class Lightbulb here.
 *
 * @author (Benedictus Kent Rachmat)
 * @version (21-9-2020)
 */
public class Goods
{
    private double value;
    private String name;

    /**
     * first constructor for objects of class Goods
     * @param a => String
     */ 
    public Goods(String a)
    {
     this.name = a;
     
    }
    
    /**
     * first constructor for objects of class Goods
     * @param a => String
     * @param b => Double
     */
    public Goods(String a, double b)
    {
     this.name = a;
     this.value = b;
    }
    
    /**
     * Get value of 'value'
     */
    public double getValue()
    {
        return value;
        
    }
    
    /**
     * Set value of 'value'
     * @param c => Double
     */
    public void setValue(double c)
    {
        this.value = c;
    }

    /**
     * Set value of 'name'
     * @param d => String
     */
    public void setName(String d)
    {
        this.name = d;
    } 
    
    /**
     * Get value of 'name'
     */
    public String getName()
    {
        return name;
    } 
    
    /**
     * Display name and value of goods
     */
    public String toString()
    {
        return "the goods " + this.name + " costs " + this.value;
    }
    
    /**
     * Return the value including the tax
     */
    public double ttc()
    {
        return value + (0.2 * value);
    }
}
