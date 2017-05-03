

template<typename TState>
stateModelBase<TState>::stateModelBase(const int y_dim, const int x_dim, const double scale, const std::string& def_file)
: ysize(y_dim), xsize(x_dim), scale(scale), init(false), def_file(def_file)
{
	// Seed random engine with random device
	std::random_device rd;
	rand_engine.seed(rd());

}


template<typename TState>
bool stateModelBase<TState>::initialise()
{
	if(def_file.length() > 0)
		init = readFromFile(def_file);
	else
		init = false;

	if(!init)
		return false;

	return true;
}

template<typename TState>
stateModelBase<TState>::stateModelBase()
: stateModelBase(0,0,0,0.0,"")
{


}
