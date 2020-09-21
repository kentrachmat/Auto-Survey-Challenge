/**
 * Write a description of class Lightbulb here.
 *
 * @author (Benedictus Kent Rachmat)
 * @version (21-9-2020)
 */
public class Stock
{
    
    
    /**
     * first Stock Constructor
     */
    public Stock()
    {
        this.quantity = 0;
    }
    
    /**
     * second Stock Constructor
     * @param c => initial value
     */
    public Stock(int c)
    {
        this.quantity = c;
    }
    
    private int quantity;
    
    /**
     * Get the quality information
     * @return quality (int)
     */
    public int getQuantity () 
    { 
        return this.quantity;
    }
    
    /**
     * Adding the quality
     * @param a => adding value
     */
    public void add (int a) 
    { 
        this.quantity = this.quantity + a;
    }
    
    /**
     * Substract the quality
     * @param b => substracting value
     * @return substracting value
     */
    public int remove (int b) 
    { 
        
        if(this.quantity > b){
        this.quantity = this.quantity - b;
        }
        else {
        this.quantity = 0;
        }
        return b;
    }
    
    /**
     * Display quantity value
     */
    public String toString()
    {
        return "the stock’s quantity is " + this.quantity;
    }
}
